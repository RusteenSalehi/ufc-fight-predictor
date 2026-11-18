// @vitest-environment node
// The committed TypeScript types must be exactly what the committed JSON
// Schema generates. (tests/test_publish_jsonschema.py checks the schema
// against the pydantic models, so together the chain cannot drift.)
import { readFile } from 'node:fs/promises'
import { expect, test } from 'vitest'
import { generateArtifactTypes } from './artifact-types.mjs'

test('src/types/artifacts.ts matches schema/artifacts.schema.json', async () => {
  const schema = JSON.parse(
    await readFile(new URL('../schema/artifacts.schema.json', import.meta.url), 'utf8'),
  )
  const committed = await readFile(new URL('../src/types/artifacts.ts', import.meta.url), 'utf8')
  expect(committed, 'run `npm run types:gen`').toBe(await generateArtifactTypes(schema))
}, 30_000)
