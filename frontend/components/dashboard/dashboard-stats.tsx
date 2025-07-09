'use client';

import { TrendingUp, TrendingDown, DollarSign, PieChart, Target, Activity } from 'lucide-react';

export function DashboardStats() {
  const stats = [
    {
      title: 'Total Portfolio Value',
      value: '$2,847,592',
      change: '+12.5%',
      changeType: 'positive',
      icon: DollarSign,
      description: 'vs. last month',
    },
    {
      title: 'Active Assets',
      value: '24',
      change: '+3',
      changeType: 'positive',
      icon: PieChart,
      description: 'tokenized assets',
    },
    {
      title: 'Monthly Returns',
      value: '+8.4%',
      change: '+2.1%',
      changeType: 'positive',
      icon: TrendingUp,
      description: 'vs. last month',
    },
    {
      title: 'Risk Score',
      value: '6.2/10',
      change: '-0.3',
      changeType: 'negative',
      icon: Target,
      description: 'moderate risk',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {stats.map((stat, index) => {
        const Icon = stat.icon;
        return (
          <div key={index} className="bg-card p-6 rounded-lg border border-border">
            <div className="flex items-center">
              <div className="p-2 bg-stellar-100 rounded-lg">
                <Icon className="h-5 w-5 text-stellar-600" />
              </div>
              <div className="ml-4 flex-1">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-muted-foreground">
                    {stat.title}
                  </p>
                  <div className="flex items-center space-x-1">
                    {stat.changeType === 'positive' ? (
                      <TrendingUp className="h-4 w-4 text-success-600" />
                    ) : (
                      <TrendingDown className="h-4 w-4 text-error-600" />
                    )}
                    <span
                      className={`text-sm font-medium ${
                        stat.changeType === 'positive'
                          ? 'text-success-600'
                          : 'text-error-600'
                      }`}
                    >
                      {stat.change}
                    </span>
                  </div>
                </div>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <p className="text-sm text-muted-foreground">{stat.description}</p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
} 