/// <reference path="./.sst/platform/config.d.ts" />

/**
 * sample.zygot.ai: a private S3 bucket behind CloudFront, with Cloudflare in
 * front (the DNS record is proxied). Cloudflare's bot check is a custom
 * security rule on the zygot.ai zone (Managed Challenge on this hostname),
 * set in the Cloudflare dashboard, not here.
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
    const bucket = new sst.aws.Bucket('Site', { access: 'cloudfront' });

    const router = new sst.aws.Router('Cdn', {
      domain: { name: 'sample.zygot.ai', dns: sst.cloudflare.dns({ proxy: true }) },
      edge: {
        viewerRequest: {
          // A folder address opens its index.html
          injection: `
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
