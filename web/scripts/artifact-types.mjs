// Shared by gen-types.mjs and the drift test (src/types/artifacts.test.ts).
import { compile } from 'json-schema-to-typescript'

const BANNER = `/* eslint-disable */
/**
 * GENERATED FILE, do not edit.
 * Source: web/schema/artifacts.schema.json, exported from ufc.publish.schemas.
 * Regenerate with \`npm run types:gen\` (from web/).
 */`

// pydantic writes JSON Schema 2020-12; json-schema-to-typescript reads the
// draft-07 forms. Two constructs are lowered so that the types stay exact:
//   prefixItems (a tuple)               -> items as an array of schemas
//   propertyNames.enum + a value schema -> one optional property per key
function lower(node) {
  if (Array.isArray(node)) return node.map(lower)
  if (node === null || typeof node !== 'object') return node
  const out = Object.fromEntries(Object.entries(node).map(([k, v]) => [k, lower(v)]))
  if (Array.isArray(out.prefixItems)) {
    out.items = out.prefixItems
    delete out.prefixItems
  }
  const keys = out.propertyNames?.enum
  if (Array.isArray(keys) && out.additionalProperties && typeof out.additionalProperties === 'object') {
    out.properties = Object.fromEntries(keys.map((k) => [k, out.additionalProperties]))
    out.additionalProperties = false
    delete out.propertyNames
  }
  return out
}

export async function generateArtifactTypes(schema) {
  return compile(lower(schema), 'PublicArtifacts', {
    bannerComment: BANNER,
    additionalProperties: false,
    unreachableDefinitions: false,
    declareExternallyReferenced: true,
    maxItems: -1,
    style: { singleQuote: true, semi: false, printWidth: 100 },
  })
}
