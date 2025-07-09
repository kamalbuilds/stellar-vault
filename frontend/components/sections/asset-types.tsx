import { Building, Gem, Palette, FileText, Factory, Coins } from 'lucide-react';

export function AssetTypes() {
  const assetTypes = [
    {
      icon: Building,
      name: 'Real Estate',
      description: 'Commercial properties, residential complexes, REITs',
      value: '$4.2T',
      growth: '+12%'
    },
    {
      icon: Gem,
      name: 'Commodities', 
      description: 'Gold, silver, oil, agricultural products',
      value: '$2.8T',
      growth: '+8%'
    },
    {
      icon: Palette,
      name: 'Art & Collectibles',
      description: 'Fine art, vintage cars, rare collectibles',
      value: '$1.6T',
      growth: '+15%'
    },
    {
      icon: FileText,
      name: 'Bonds & Securities',
      description: 'Corporate bonds, government securities',
      value: '$3.4T',
      growth: '+6%'
    },
    {
      icon: Factory,
      name: 'Infrastructure',
      description: 'Solar farms, wind turbines, transportation',
      value: '$2.1T',
      growth: '+18%'
    },
    {
      icon: Coins,
      name: 'Precious Metals',
      description: 'Gold bars, silver coins, platinum',
      value: '$1.9T',
      growth: '+10%'
    }
  ];

  return (
    <section className="py-16 lg:py-24 bg-muted/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
            Tokenize Any Real-World Asset
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Our platform supports a wide range of asset classes, making previously illiquid investments accessible to everyone
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {assetTypes.map((asset, index) => {
            const Icon = asset.icon;
            return (
              <div key={index} className="group p-6 bg-card border border-border rounded-xl hover:border-stellar-300 hover:shadow-lg transition-all card-hover">
                <div className="flex items-start justify-between mb-4">
                  <div className="w-12 h-12 bg-stellar-100 rounded-lg flex items-center justify-center group-hover:bg-stellar-200 transition-colors">
                    <Icon className="w-6 h-6 text-stellar-600" />
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-muted-foreground">Market Size</div>
                    <div className="text-lg font-bold text-foreground">{asset.value}</div>
                  </div>
                </div>
                
                <h3 className="text-xl font-semibold text-foreground mb-2">
                  {asset.name}
                </h3>
                
                <p className="text-muted-foreground mb-4 text-sm">
                  {asset.description}
                </p>
                
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">YoY Growth</span>
                  <span className="text-sm font-medium text-success-600">{asset.growth}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-12 text-center">
          <p className="text-muted-foreground mb-4">
            Combined market opportunity: <span className="font-bold text-foreground">$16+ Trillion</span>
          </p>
          <button className="inline-flex items-center px-6 py-3 bg-stellar-600 text-white font-medium rounded-lg hover:bg-stellar-700 transition-colors">
            Explore All Asset Types
          </button>
        </div>
      </div>
    </section>
  );
} 