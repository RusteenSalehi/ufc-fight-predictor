// @vitest-environment node
import path from 'node:path'
import { describe, expect, test } from 'vitest'
import { resolveDataFile } from './vite.publish-data'

const root = path.resolve('/srv/publish/public')

describe('resolveDataFile', () => {
  test('maps /data/ paths to JSON files under the root', () => {
    expect(resolveDataFile(root, '/data/upcoming.json')).toBe(path.join(root, 'upcoming.json'))
    expect(resolveDataFile(root, '/data/bouts/wb0000000000000001.json?x=1')).toBe(
      path.join(root, 'bouts', 'wb0000000000000001.json'),
    )
  })

  test('refuses anything outside the root or not JSON', () => {
    expect(resolveDataFile(root, '/data/../archive/x.json')).toBeNull()
    expect(resolveDataFile(root, '/data/%2e%2e/%2e%2e/etc/passwd.json')).toBeNull()
    expect(resolveDataFile(root, '/data/upcoming.txt')).toBeNull()
    expect(resolveDataFile(root, '/data/%E0%A4%A.json')).toBeNull()
    expect(resolveDataFile(root, '/src/main.tsx')).toBeNull()
  })
})
