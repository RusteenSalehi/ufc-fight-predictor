/** Site-wide constants. The name is a placeholder until branding is chosen
 * (SPEC: custom domain and branding are deferred). */
export const SITE_NAME = 'Bout Model'

export const REPO_URL = 'https://github.com/RusteenSalehi/ufc-fight-predictor'

/** Where the published JSON lives. In development `vite` serves
 * ../data/publish/public at /data/ (vite.publish-data.ts); a deployment sets
 * VITE_DATA_BASE_URL at build time if the files live elsewhere. */
export const DATA_BASE_URL: string = (import.meta.env.VITE_DATA_BASE_URL ?? '/data').replace(
  /\/+$/,
  '',
)

export const DISCLAIMER =
  'Not affiliated with the UFC. Predictions are statistical estimates, not betting advice.'
