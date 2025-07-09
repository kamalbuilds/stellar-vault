'use client';

import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

export function MarketTrends() {
  const trends = [
    {
      asset: 'Real Estate',
      price: '$2,450/token',
      change: '+5.2%',
      changeType: 'positive',
      volume: '$1.2M',
    },
    {
      asset: 'Gold',
      price: '$1,985/oz',
      change: '-1.8%',
      changeType: 'negative',
      volume: '$850K',
    },
    {
      asset: 'Art Index',
      price: '$450/share',
      change: '+8.7%',
      changeType: 'positive',
      volume: '$320K',
    },
    {
      asset: 'Solar Energy',
      price: '$89/token',
      change: '+12.3%',
      changeType: 'positive',
      volume: '$2.1M',
    },
    {
      asset: 'Corporate Bonds',
      price: '$102.50',
      change: '+0.3%',
      changeType: 'positive',
      volume: '$1.8M',
    },
  ];

  const topMovers = [
    { name: 'Tesla Gigafactory', change: '+23.5%', type: 'positive' },
    { name: 'Vineyard Estate', change: '+15.2%', type: 'positive' },
    { name: 'Lithium Mining', change: '-8.1%', type: 'negative' },
  ];

  return (
    <div className="bg-card p-6 rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Market Trends</h3>
          <p className="text-sm text-muted-foreground">Live market data</p>
        </div>
        <a
          href="/markets"
          className="text-sm text-stellar-600 hover:text-stellar-700 font-medium"
        >
          View Markets
        </a>
      </div>

      <div className="space-y-4">
        {trends.map((trend, index) => (
          <div key={index} className="flex items-center justify-between p-3 rounded-lg hover:bg-muted transition-colors">
            <div className="flex-1">
              <div className="font-medium text-foreground">{trend.asset}</div>
              <div className="text-sm text-muted-foreground">Vol: {trend.volume}</div>
            </div>
            <div className="text-right">
              <div className="font-medium text-foreground">{trend.price}</div>
              <div className={`text-sm font-medium flex items-center ${
                trend.changeType === 'positive' ? 'text-success-600' : 'text-error-600'
              }`}>
                {trend.changeType === 'positive' ? (
                  <TrendingUp className="w-3 h-3 mr-1" />
                ) : (
                  <TrendingDown className="w-3 h-3 mr-1" />
                )}
                {trend.change}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border">
        <h4 className="font-medium text-foreground mb-3 flex items-center">
          <Activity className="w-4 h-4 mr-2" />
          Top Movers
        </h4>
        <div className="space-y-2">
          {topMovers.map((mover, index) => (
            <div key={index} className="flex items-center justify-between text-sm">
              <span className="text-foreground">{mover.name}</span>
              <span className={`font-medium ${
                mover.type === 'positive' ? 'text-success-600' : 'text-error-600'
              }`}>
                {mover.change}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 