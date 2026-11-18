import { Link } from 'react-router'
import { usePageTitle } from '../usePageTitle'
import { EmptyState } from '../components/StatusViews'

export function NotFound() {
  usePageTitle('Not found')
  return (
    <EmptyState title="Page not found">
      There is nothing at this address.{' '}
      <Link to="/" className="link">
        See upcoming events
      </Link>
      .
    </EmptyState>
  )
}
