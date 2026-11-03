/// <reference path="./.sst/platform/config.d.ts" />

/**
 * sample.zygot.ai: a private S3 bucket behind CloudFront, with a browser
 * sign-in (one password, any username) checked at the edge.
 *
 * SST only makes the bucket and the site's address; `ego-rerun publish`
 * uploads the episodes, the Rerun viewer and the page into the bucket.
 */
export default $config({
  app() {
    return {
      name: 'zygot-sample',
      home: 'aws',
      removal: 'remove',
      providers: {
        aws: {
          region: process.env.AWS_REGION as 'ap-south-1',
          allowedAccountIds: [process.env.AWS_ACCOUNT_ID!],
        },
        cloudflare: '6.15.0',
      },
    };
  },
  async run() {
    const password = process.env.SITE_PASSWORD;
    if (!password) throw new Error('SITE_PASSWORD is missing from .env.production.local');

    const bucket = new sst.aws.Bucket('Site', { access: 'cloudfront' });

    const router = new sst.aws.Router('Cdn', {
      domain: { name: 'sample.zygot.ai', dns: sst.cloudflare.dns() },
      edge: {
        viewerRequest: {
          // Sign-in first; then a folder address opens its index.html
          injection: `
  var auth = event.request.headers.authorization;
  var pair = auth && auth.value.indexOf('Basic ') === 0 ? Buffer.from(auth.value.slice(6), 'base64').toString() : '';
  if (pair.slice(pair.indexOf(':') + 1) !== ${JSON.stringify(password)}) {
    return {
      statusCode: 401,
      statusDescription: 'Unauthorized',
      headers: { 'www-authenticate': { value: 'Basic realm="Zygot sample", charset="UTF-8"' } },
    };
  }
  if (event.request.uri.endsWith('/')) event.request.uri += 'index.html';`,
        },
      },
    });
    router.routeBucket('/', bucket);

    return {
      url: router.url,
      bucket: bucket.name,
      distribution: router.distributionID,
    };
  },
});
