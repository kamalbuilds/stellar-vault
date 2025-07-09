import { Shield, Brain, TrendingUp, Globe, Lock, Zap } from 'lucide-react';

export function Features() {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Valuation',
      description: 'Machine learning algorithms provide accurate, real-time asset valuations using market data, comparable sales, and predictive analytics.',
      highlights: ['95%+ accuracy', 'Real-time updates', 'Multi-asset support']
    },
    {
      icon: Shield,
      title: 'Regulatory Compliance',
      description: 'Built-in KYC/AML processes and automated compliance monitoring ensure adherence to global financial regulations.',
      highlights: ['SEC compliant', 'Automated KYC', 'Audit trails']
    },
    {
      icon: TrendingUp,
      title: 'Smart Portfolio Management',
      description: 'AI-driven portfolio optimization and risk assessment help maximize returns while maintaining optimal diversification.',
      highlights: ['Risk optimization', 'Auto-rebalancing', 'Performance analytics']
    },
    {
      icon: Globe,
      title: 'Global Market Access',
      description: 'Trade tokenized assets 24/7 on our global marketplace with instant settlement and cross-border capabilities.',
      highlights: ['24/7 trading', 'Global reach', '3-5s settlement']
    },
    {
      icon: Lock,
      title: 'Enterprise Security',
      description: 'Multi-signature wallets, hardware security modules, and institutional-grade custody solutions protect your assets.',
      highlights: ['Multi-sig wallets', 'HSM integration', 'Insurance coverage']
    },
    {
      icon: Zap,
      title: 'Low-Cost Transactions',
      description: 'Leverage Stellar\'s ultra-low transaction fees to make micro-investments and frequent trading economically viable.',
      highlights: ['$0.00000392 fees', 'Micro-investing', 'High throughput']
    }
  ];

  return (
    <section className="py-16 lg:py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
            Revolutionary Features for RWA Tokenization
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Combining cutting-edge AI with blockchain technology to democratize access to real-world assets
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div key={index} className="group p-6 bg-card border border-border rounded-xl hover:shadow-lg transition-shadow">
                <div className="w-12 h-12 bg-stellar-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-stellar-200 transition-colors">
                  <Icon className="w-6 h-6 text-stellar-600" />
                </div>
                
                <h3 className="text-xl font-semibold text-foreground mb-3">
                  {feature.title}
                </h3>
                
                <p className="text-muted-foreground mb-4">
                  {feature.description}
                </p>
                
                <ul className="space-y-2">
                  {feature.highlights.map((highlight, i) => (
                    <li key={i} className="flex items-center text-sm text-muted-foreground">
                      <div className="w-1.5 h-1.5 bg-stellar-600 rounded-full mr-3"></div>
                      {highlight}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>

        {/* Bottom CTA */}
        <div className="mt-16 text-center">
          <div className="inline-flex items-center px-6 py-3 bg-muted rounded-full text-muted-foreground text-sm">
            <Shield className="w-4 h-4 mr-2" />
            Trusted by institutions worldwide
          </div>
        </div>
      </div>
    </section>
  );
} 