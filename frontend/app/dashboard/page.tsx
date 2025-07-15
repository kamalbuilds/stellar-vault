'use client';

import WalletConnect from '@/components/WalletConnect';
import { useWallet, WalletStatus } from '@/contexts/WalletContext';

export default function DashboardPage() {
  const { status, account, error } = useWallet();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
          <p className="text-gray-600 dark:text-gray-300 mt-2">
            Connect your Stellar wallet to access tokenization features
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Wallet Section */}
          <div className="lg:col-span-1">
            <WalletConnect />
            
            {/* Debug Info */}
            <div className="mt-6 p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold mb-2 text-gray-900 dark:text-white">Debug Info</h3>
              <div className="text-sm space-y-1 text-gray-600 dark:text-gray-300">
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Account:</strong> {account ? 'Connected' : 'None'}</p>
                <p><strong>Public Key:</strong> {account?.publicKey ? `${account.publicKey.slice(0, 12)}...` : 'None'}</p>
                <p><strong>Network:</strong> {account?.network || 'None'}</p>
                <p><strong>Error:</strong> {error || 'None'}</p>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2">
            {status === WalletStatus.CONNECTED ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Welcome to StellarVault</h2>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  Your wallet is connected. You can now start tokenizing assets.
                </p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <button className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 text-left">
                    <h3 className="font-medium mb-2 text-gray-900 dark:text-white">Tokenize Real Estate</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">Convert property into tradeable tokens</p>
                  </button>
                  
                  <button className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 text-left">
                    <h3 className="font-medium mb-2 text-gray-900 dark:text-white">Tokenize Commodities</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">Create tokens for physical goods</p>
                  </button>
                  
                  <button className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 text-left">
                    <h3 className="font-medium mb-2 text-gray-900 dark:text-white">Tokenize Art</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">Fractionalize art ownership</p>
                  </button>
                  
                  <button className="p-4 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 text-left">
                    <h3 className="font-medium mb-2 text-gray-900 dark:text-white">View Portfolio</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">Track your tokenized assets</p>
                  </button>
                </div>
              </div>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 text-center">
                <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">Connect Your Wallet</h2>
                <p className="text-gray-600 dark:text-gray-300">
                  Please connect your Stellar wallet to access the platform features.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 