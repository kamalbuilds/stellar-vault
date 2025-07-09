'use client';

import { ArrowUpRight, ArrowDownLeft, ArrowRightLeft, Clock } from 'lucide-react';

export function RecentTransactions() {
  const transactions = [
    {
      id: '1',
      type: 'buy',
      asset: 'Manhattan Office Building',
      amount: '+$25,000',
      time: '2 hours ago',
      status: 'completed',
    },
    {
      id: '2',
      type: 'sell',
      asset: 'Gold ETF Tokens',
      amount: '-$15,500',
      time: '4 hours ago',
      status: 'completed',
    },
    {
      id: '3',
      type: 'transfer',
      asset: 'Art Collection NFT',
      amount: '$8,200',
      time: '1 day ago',
      status: 'pending',
    },
    {
      id: '4',
      type: 'buy',
      asset: 'Tesla Solar Farm',
      amount: '+$45,000',
      time: '2 days ago',
      status: 'completed',
    },
    {
      id: '5',
      type: 'sell',
      asset: 'Corporate Bonds',
      amount: '-$30,000',
      time: '3 days ago',
      status: 'completed',
    },
  ];

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'buy':
        return <ArrowDownLeft className="w-4 h-4 text-success-600" />;
      case 'sell':
        return <ArrowUpRight className="w-4 h-4 text-error-600" />;
      case 'transfer':
        return <ArrowRightLeft className="w-4 h-4 text-blue-600" />;
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getAmountColor = (type: string) => {
    switch (type) {
      case 'buy':
        return 'text-success-600';
      case 'sell':
        return 'text-error-600';
      case 'transfer':
        return 'text-blue-600';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <div className="bg-card p-6 rounded-lg border border-border">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-foreground">Recent Transactions</h3>
          <p className="text-sm text-muted-foreground">Latest portfolio activity</p>
        </div>
        <a
          href="/transactions"
          className="text-sm text-stellar-600 hover:text-stellar-700 font-medium"
        >
          View All
        </a>
      </div>

      <div className="space-y-4">
        {transactions.map((transaction) => (
          <div key={transaction.id} className="flex items-center space-x-4 p-3 rounded-lg hover:bg-muted transition-colors">
            <div className="p-2 bg-muted rounded-lg">
              {getTransactionIcon(transaction.type)}
            </div>
            
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <div className="font-medium text-foreground truncate">
                  {transaction.asset}
                </div>
                <div className={`font-semibold ${getAmountColor(transaction.type)}`}>
                  {transaction.amount}
                </div>
              </div>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <div className="capitalize">{transaction.type}</div>
                <div>{transaction.time}</div>
              </div>
            </div>
            
            <div className="flex items-center">
              <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                transaction.status === 'completed' 
                  ? 'bg-success-100 text-success-800 dark:bg-success-900/20 dark:text-success-300'
                  : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300'
              }`}>
                {transaction.status}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 