/**
 * Runs the SST CLI signed in to the sample site's account only:
 * the keys come from .env.production.local (loaded by node --env-file),
 * and any shell profile or session is dropped.
 */
import { spawnSync } from 'node:child_process';

const env = { ...process.env };
delete env.AWS_PROFILE;
delete env.AWS_SESSION_TOKEN;

const { status } = spawnSync('pnpm', ['exec', 'sst', ...process.argv.slice(2)], { stdio: 'inherit', env });
process.exitCode = status ?? 1;
