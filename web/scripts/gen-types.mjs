// Generate src/types/artifacts.ts from the committed JSON Schema, which
// `uv run python -m ufc.publish schema` writes from the pydantic models.
//   node scripts/gen-types.mjs           write the file
//   node scripts/gen-types.mjs --check   exit 1 if the file is out of date
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { generateArtifactTypes } from './artifact-types.mjs'

const schemaPath = fileURLToPath(new URL('../schema/artifacts.schema.json', import.meta.url))
const outPath = fileURLToPath(new URL('../src/types/artifacts.ts', import.meta.url))

const schema = JSON.parse(await readFile(schemaPath, 'utf8'))
const expected = await generateArtifactTypes(schema)

if (process.argv.includes('--check')) {
  let current = null
  try {
    current = await readFile(outPath, 'utf8')
  } catch {
    // missing file is drift too
  }
  if (current !== expected) {
    console.error('src/types/artifacts.ts is out of date; run `npm run types:gen`')
    process.exit(1)
  }
  console.log('src/types/artifacts.ts matches schema/artifacts.schema.json')
} else {
  await writeFile(outPath, expected)
  console.log('wrote src/types/artifacts.ts')
}
