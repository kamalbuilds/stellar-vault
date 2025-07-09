'use client';

import { useState } from 'react';
import { Plus, TrendingUp, ArrowRightLeft, Settings, ChevronDown } from 'lucide-react';

export function QuickActions() {
  const [isOpen, setIsOpen] = useState(false);

  const actions = [
    {
      icon: Plus,
      label: 'Tokenize Asset',
      description: 'Create a new tokenized asset',
      href: '/tokenize',
      color: 'bg-stellar-600 hover:bg-stellar-700',
    },
    {
      icon: TrendingUp,
      label: 'Trade Assets',
      description: 'Buy or sell tokenized assets',
      href: '/trade',
      color: 'bg-vault-600 hover:bg-vault-700',
    },
    {
      icon: ArrowRightLeft,
      label: 'Transfer',
      description: 'Transfer assets to another wallet',
      href: '/transfer',
      color: 'bg-success-600 hover:bg-success-700',
    },
  ];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center px-4 py-2 bg-stellar-600 text-white font-medium rounded-lg hover:bg-stellar-700 transition-colors"
      >
        <Plus className="w-4 h-4 mr-2" />
        Quick Actions
        <ChevronDown className="w-4 h-4 ml-2" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-card border border-border rounded-lg shadow-lg z-50">
          <div className="p-2">
            {actions.map((action, index) => {
              const Icon = action.icon;
              return (
                <a
                  key={index}
                  href={action.href}
                  className="flex items-center p-3 rounded-lg hover:bg-muted transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  <div className={`p-2 rounded-lg ${action.color} mr-3`}>
                    <Icon className="w-4 h-4 text-white" />
                  </div>
                  <div>
                    <div className="font-medium text-foreground">{action.label}</div>
                    <div className="text-sm text-muted-foreground">{action.description}</div>
                  </div>
                </a>
              );
            })}
          </div>
          
          <div className="border-t border-border p-2">
            <a
              href="/settings"
              className="flex items-center p-3 rounded-lg hover:bg-muted transition-colors"
              onClick={() => setIsOpen(false)}
            >
              <div className="p-2 rounded-lg bg-muted mr-3">
                <Settings className="w-4 h-4 text-muted-foreground" />
              </div>
              <div>
                <div className="font-medium text-foreground">Settings</div>
                <div className="text-sm text-muted-foreground">Manage your account</div>
              </div>
            </a>
          </div>
        </div>
      )}
    </div>
  );
} 