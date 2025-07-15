// stellar-vault/frontend/components/WalletConnect.tsx
'use client';

import { useState } from 'react';
import { useWallet, WalletStatus } from '../contexts/WalletContext';
import { StellarNetwork } from '../lib/stellar';

// Wallet selection modal component
function WalletSelectionModal({ 
  isOpen, 
  onClose, 
  onWalletSelect 
}: {
  isOpen: boolean;
  onClose: () => void;
  onWalletSelect: (walletId: string) => void;
}) {
  const { supportedWallets } = useWallet();

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 border border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Connect Wallet</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300"
          >
            ×
          </button>
        </div>
        
        <div className="space-y-3">
          {supportedWallets.map((wallet) => (
            <button
              key={wallet.id}
              onClick={() => onWalletSelect(wallet.id)}
              className="w-full flex items-center space-x-3 p-3 rounded-lg border transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 border-gray-200 dark:border-gray-600"
            >
              <img
                src={wallet.icon}
                alt={wallet.name}
                className="w-8 h-8"
              />
              <div className="flex-1 text-left">
                <p className="font-medium text-gray-900 dark:text-white">{wallet.name}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">Click to connect</p>
              </div>
              <span className="text-blue-600 dark:text-blue-400 text-sm">Connect</span>
            </button>
          ))}
        </div>
        
        <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 text-center">
          Don't have a wallet? Install one from the options above.
        </div>
      </div>
    </div>
  );
}

// Network selector component
function NetworkSelector() {
  const { account, switchNetwork } = useWallet();
  const [isOpen, setIsOpen] = useState(false);

  const networks = [
    { value: StellarNetwork.TESTNET, label: 'Testnet', color: 'bg-yellow-100 text-yellow-800' },
    { value: StellarNetwork.FUTURENET, label: 'Futurenet', color: 'bg-purple-100 text-purple-800' },
    { value: StellarNetwork.MAINNET, label: 'Mainnet', color: 'bg-green-100 text-green-800' }
  ];

  const currentNetwork = networks.find(n => n.value === account?.network);

  const handleNetworkChange = async (network: StellarNetwork) => {
    try {
      await switchNetwork(network);
      setIsOpen(false);
    } catch (error) {
      console.error('Failed to switch network:', error);
    }
  };

  if (!account) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`px-3 py-1 rounded-full text-xs font-medium ${currentNetwork?.color || 'bg-gray-100'}`}
      >
        {currentNetwork?.label || 'Unknown Network'}
      </button>
      
      {isOpen && (
        <div className="absolute top-full right-0 mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg py-1 z-10">
          {networks.map((network) => (
            <button
              key={network.value}
              onClick={() => handleNetworkChange(network.value)}
              className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-white"
            >
              <span className={`inline-block w-2 h-2 rounded-full mr-2 ${network.color.split(' ')[0]}`}></span>
              {network.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Account details component
function AccountDetails() {
  const { account, formatAmount, refreshAccount } = useWallet();
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await refreshAccount();
    } finally {
      setIsRefreshing(false);
    }
  };

  if (!account) return null;

  return (
    <div className="space-y-4">
      {/* Account Info */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium text-gray-900 dark:text-white">Account</h4>
          <div className="flex items-center space-x-2">
            <NetworkSelector />
            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
              title="Refresh account data"
            >
              <svg 
                className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>
        
        <div className="space-y-2">
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Public Key</p>
            <p className="text-sm font-mono break-all text-gray-900 dark:text-white">
              {`${account.publicKey.slice(0, 6)}...${account.publicKey.slice(-6)}`}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 dark:text-gray-400">Wallet</p>
            <p className="text-sm text-gray-900 dark:text-white">{account.walletType}</p>
          </div>
        </div>
      </div>

      {/* Balances */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 dark:text-white mb-3">Balances</h4>
        <div className="space-y-2">
          {account.balances.length > 0 ? (
            account.balances.map((balance, index) => (
              <div key={index} className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-300">{balance.code}</span>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {formatAmount(balance.balance)} {balance.code}
                </span>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">No assets found</p>
          )}
        </div>
      </div>

      {/* Recent Transactions */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 dark:text-white mb-3">Recent Transactions</h4>
        <div className="space-y-2">
          {account.transactions.length > 0 ? (
            account.transactions.slice(0, 3).map((tx) => (
              <div key={tx.id} className="flex justify-between items-center text-sm">
                <div>
                  <p className="font-medium capitalize text-gray-900 dark:text-white">{tx.type}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(tx.timestamp).toLocaleDateString()}
                  </p>
                </div>
                <div className="text-right">
                  {tx.amount && (
                    <p className="font-medium text-gray-900 dark:text-white">
                      {formatAmount(tx.amount)} {tx.asset}
                    </p>
                  )}
                  <p className={`text-xs ${tx.successful ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                    {tx.successful ? 'Success' : 'Failed'}
                  </p>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400">No transactions found</p>
          )}
        </div>
      </div>
    </div>
  );
}

// Main wallet connect component
export default function WalletConnect() {
  const { status, account, error, connect, disconnect } = useWallet();
  const [showWalletModal, setShowWalletModal] = useState(false);

  const handleConnect = () => {
    setShowWalletModal(true);
  };

  const handleWalletSelect = async (walletId: string) => {
    setShowWalletModal(false);
    try {
      await connect(walletId);
    } catch (error) {
      console.error('Failed to connect:', error);
    }
  };

  const handleDisconnect = () => {
    disconnect();
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
        <div className="p-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Stellar Wallet</h3>
          
          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </div>
          )}
          
          {/* Connection Status */}
          {status === WalletStatus.DISCONNECTED && (
            <div className="text-center">
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                Connect your Stellar wallet to access StellarVault features
              </p>
              <button
                onClick={handleConnect}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors"
              >
                Connect Wallet
              </button>
            </div>
          )}
          
          {status === WalletStatus.CONNECTING && (
            <div className="text-center">
              <div className="inline-flex items-center space-x-2 text-blue-600 dark:text-blue-400">
                <svg className="animate-spin w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Connecting...</span>
              </div>
            </div>
          )}
          
          {status === WalletStatus.CONNECTED && account && (
            <div>
              <AccountDetails />
              <button
                onClick={handleDisconnect}
                className="w-full mt-4 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200 py-2 px-4 rounded-lg font-medium hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                Disconnect Wallet
              </button>
            </div>
          )}
          
          {status === WalletStatus.ERROR && (
            <div className="text-center">
              <p className="text-red-600 dark:text-red-400 mb-4">Failed to connect wallet</p>
              <button
                onClick={handleConnect}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Wallet Selection Modal */}
      <WalletSelectionModal
        isOpen={showWalletModal}
        onClose={() => setShowWalletModal(false)}
        onWalletSelect={handleWalletSelect}
      />
    </div>
  );
}