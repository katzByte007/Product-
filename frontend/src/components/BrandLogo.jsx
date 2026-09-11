export default function BrandLogo({ variant = 'header' }) {
  return (
    <div className={`brand-logo ${variant}`}>
      <div className="brand-icon">👁</div>
      <div className="brand-text">
        <span className="brand-name">Vision AI</span>
        {variant === 'login' && <span className="brand-tag">Plant monitoring & analytics</span>}
      </div>
    </div>
  );
}
