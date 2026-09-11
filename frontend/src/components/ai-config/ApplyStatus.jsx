export default function ApplyStatus({ message, type }) {
  if (!message) return null;
  return <div className={`apply-status ${type || ''}`}>{message}</div>;
}
