'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export function PortfolioOverview() {
  const data = [
    { name: 'Jan', value: 2400000 },
    { name: 'Feb', value: 2100000 },
    { name: 'Mar', value: 2300000 },
    { name: 'Apr', value: 2650000 },
    { name: 'May', value: 2450000 },
    { name: 'Jun', value: 2847592 },
  ];

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  return (
    <div className="bg-card p-6 rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Portfolio Performance</h3>
          <p className="text-sm text-muted-foreground">6-month trend</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-foreground">$2,847,592</div>
          <div className="text-sm text-success-600">+12.5% this month</div>
        </div>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis 
              dataKey="name" 
              stroke="#6b7280"
              fontSize={12}
            />
            <YAxis 
              stroke="#6b7280"
              fontSize={12}
              tickFormatter={formatCurrency}
            />
            <Tooltip 
              formatter={(value) => [formatCurrency(value as number), 'Portfolio Value']}
              labelStyle={{ color: '#374151' }}
              contentStyle={{ 
                backgroundColor: '#ffffff', 
                border: '1px solid #e5e7eb',
                borderRadius: '8px'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#0891b2" 
              strokeWidth={2}
              dot={{ fill: '#0891b2', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#0891b2', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="text-center p-4 bg-muted rounded-lg">
          <div className="text-lg font-semibold text-foreground">24</div>
          <div className="text-sm text-muted-foreground">Total Assets</div>
        </div>
        <div className="text-center p-4 bg-muted rounded-lg">
          <div className="text-lg font-semibold text-foreground">$118,650</div>
          <div className="text-sm text-muted-foreground">Monthly Income</div>
        </div>
        <div className="text-center p-4 bg-muted rounded-lg">
          <div className="text-lg font-semibold text-foreground">8.4%</div>
          <div className="text-sm text-muted-foreground">Annual Return</div>
        </div>
      </div>
    </div>
  );
} 