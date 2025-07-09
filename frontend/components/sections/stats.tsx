export function Stats() {
  const stats = [
    {
      value: '$16T',
      label: 'RWA Market Size',
      description: 'Total addressable market by 2030'
    },
    {
      value: '0.00000392',
      label: 'Transaction Cost',
      description: 'Ultra-low fees on Stellar'
    },
    {
      value: '95%+',
      label: 'AI Accuracy',
      description: 'Intelligent asset valuation'
    },
    {
      value: '24/7',
      label: 'Global Trading',
      description: 'Continuous market access'
    }
  ];

  return (
    <section className="py-16 lg:py-24 bg-muted/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
            Transforming Asset Investment
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Built for the future of finance with institutional-grade security and performance
          </p>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl lg:text-4xl font-bold text-stellar-600 mb-2">
                {stat.value}
              </div>
              <div className="text-lg font-semibold text-foreground mb-1">
                {stat.label}
              </div>
              <div className="text-sm text-muted-foreground">
                {stat.description}
              </div>
            </div>
          ))}
        </div>

        {/* Key features grid */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div className="p-6 bg-card border border-border rounded-xl">
            <div className="w-12 h-12 bg-stellar-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-stellar-600 font-bold">AI</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Intelligent Valuation
            </h3>
            <p className="text-muted-foreground">
              ML-powered asset pricing with real-time market data and predictive analytics
            </p>
          </div>

          <div className="p-6 bg-card border border-border rounded-xl">
            <div className="w-12 h-12 bg-vault-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-vault-600 font-bold">$</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Fractional Ownership
            </h3>
            <p className="text-muted-foreground">
              Make high-value assets accessible through blockchain-based tokenization
            </p>
          </div>

          <div className="p-6 bg-card border border-border rounded-xl">
            <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-success-600 font-bold">⚡</span>
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">
              Instant Settlement
            </h3>
            <p className="text-muted-foreground">
              3-5 second transaction finality on Stellar's high-performance network
            </p>
          </div>
        </div>
      </div>
    </section>
  );
} 