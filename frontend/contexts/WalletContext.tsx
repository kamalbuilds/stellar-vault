// stellar-vault/frontend/contexts/WalletContext.tsx
'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode, useMemo } from 'react';
import {
  StellarWalletsKit,
  WalletNetwork,
  allowAllModules,
  ISupportedWallet,
  FREIGHTER_ID,
  ALBEDO_ID,
  XBULL_ID,
} from '@creit.tech/stellar-wallets-kit';
import { StellarService, StellarNetwork, AssetInfo, TransactionInfo } from '../lib/stellar';

// Wallet connection status
export enum WalletStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error'
}

// Wallet account information
export interface WalletAccount {
  publicKey: string;
  network: StellarNetwork;
  walletType: string;
  balances: AssetInfo[];
  transactions: TransactionInfo[];
}

// Wallet context state
export interface WalletContextState {
  // Connection state
  status: WalletStatus;
  account: WalletAccount | null;
  error: string | null;
  
  // Available wallets
  supportedWallets: ISupportedWallet[];
  
  // Connection methods
  connect: (walletId?: string) => Promise<void>;
  disconnect: () => void;
  
  // Account operations
  refreshAccount: () => Promise<void>;
  switchNetwork: (network: StellarNetwork) => Promise<void>;
  
  // Transaction methods
  sendPayment: (params: {
    destinationKey: string;
    amount: string;
    assetCode?: string;
    memo?: string;
  }) => Promise<any>;
  
  signTransaction: (xdr: string) => Promise<string>;
  
  // Asset operations
  createAsset: (params: {
    assetCode: string;
    limit?: string;
  }) => Promise<any>;
  
  tokenizeAsset: (params: {
    assetType: 'real_estate' | 'commodity' | 'art' | 'bond';
    assetName: string;
    totalSupply: string;
    description: string;
    metadata: Record<string, any>;
  }) => Promise<any>;
  
  // Portfolio management
  getPortfolioValue: () => Promise<{
    totalValue: string;
    assets: Array<AssetInfo & { value: string; percentage: number }>;
  }>;
  
  // Utility methods
  isValidAddress: (address: string) => boolean;
  formatAmount: (amount: string, decimals?: number) => string;
}

// Create context
const WalletContext = createContext<WalletContextState | null>(null);

// Wallet provider props
export interface WalletProviderProps {
  children: ReactNode;
  defaultNetwork?: StellarNetwork;
  enabledWallets?: string[];
  autoConnect?: boolean;
}

// Wallet provider component
export function WalletProvider({
  children,
  defaultNetwork = StellarNetwork.TESTNET,
  enabledWallets = [FREIGHTER_ID, ALBEDO_ID, XBULL_ID],
  autoConnect = true
}: WalletProviderProps) {
  // Memoize enabled wallets to prevent re-initialization
  const memoizedEnabledWallets = useMemo(() => enabledWallets, [enabledWallets]);

  // State management
  const [status, setStatus] = useState<WalletStatus>(WalletStatus.DISCONNECTED);
  const [account, setAccount] = useState<WalletAccount | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [walletKit, setWalletKit] = useState<StellarWalletsKit | null>(null);
  const [supportedWallets, setSupportedWallets] = useState<ISupportedWallet[]>([]);
  const [currentWalletId, setCurrentWalletId] = useState<string | null>(null);

  // Initialize wallet kit
  useEffect(() => {
    // Prevent re-initialization if already initialized
    if (walletKit) {
      return;
    }

    const initializeWalletKit = async () => {
      try {
        const kit = new StellarWalletsKit({
          network: defaultNetwork === StellarNetwork.MAINNET ? WalletNetwork.PUBLIC : WalletNetwork.TESTNET,
          selectedWalletId: FREIGHTER_ID,
          modules: allowAllModules()
        });

        setWalletKit(kit);
        
        // Get supported wallets
        const wallets = await kit.getSupportedWallets();
        const filteredWallets = wallets.filter((wallet: ISupportedWallet) => 
          memoizedEnabledWallets.includes(wallet.id)
        );
        setSupportedWallets(filteredWallets);
        console.log('WalletKit initialized with wallets:', filteredWallets.map(w => w.name));

        // Set network in StellarService
        StellarService.setNetwork(defaultNetwork);
      } catch (err: any) {
        console.error('Failed to initialize wallet kit:', err);
        setError('Failed to initialize wallet connection');
      }
    };

    initializeWalletKit();
  }, [defaultNetwork, memoizedEnabledWallets, walletKit]);

  // Auto-connect if enabled
  useEffect(() => {
    if (autoConnect && walletKit && status === WalletStatus.DISCONNECTED) {
      // Check if there's a previously connected wallet
      const lastWalletId = localStorage.getItem('stellarvault_last_wallet');
      if (lastWalletId) {
        connect(lastWalletId);
      }
    }
  }, [walletKit, autoConnect]);

  // Connect to wallet
  const connect = useCallback(async (walletId?: string) => {
    console.log('Connect function called with walletId:', walletId);
    
    if (!walletKit) {
      console.error('Wallet kit not initialized!');
      setError('Wallet kit not initialized');
      return;
    }

    setStatus(WalletStatus.CONNECTING);
    setError(null);

    try {
      if (walletId) {
        walletKit.setWallet(walletId);
        setCurrentWalletId(walletId);
      }

      // Get wallet address
      const { address } = await walletKit.getAddress();
      
      if (!address) {
        throw new Error('No address returned from wallet');
      }

      // Get account information
      const balances = await StellarService.getAccountBalance(address);
      const transactions = await StellarService.getAccountTransactions(address, 5);

      // Get current wallet info
      const wallets = await walletKit.getSupportedWallets();
      const currentWallet = wallets.find((w: ISupportedWallet) => w.id === currentWalletId);

      const walletAccount: WalletAccount = {
        publicKey: address,
        network: StellarService.getCurrentNetwork(),
        walletType: currentWallet?.name || 'Unknown',
        balances,
        transactions
      };

      setAccount(walletAccount);
      setStatus(WalletStatus.CONNECTED);

      // Store last connected wallet
      localStorage.setItem('stellarvault_last_wallet', currentWalletId || '');
      
      console.log('Wallet connected successfully:', walletAccount);
    } catch (err: any) {
      console.error('Failed to connect wallet:', err);
      setError(err.message || 'Failed to connect wallet');
      setStatus(WalletStatus.ERROR);
    }
  }, [walletKit, currentWalletId]);

  // Disconnect wallet
  const disconnect = useCallback(() => {
    setAccount(null);
    setStatus(WalletStatus.DISCONNECTED);
    setError(null);
    setCurrentWalletId(null);
    localStorage.removeItem('stellarvault_last_wallet');
    console.log('Wallet disconnected');
  }, []);

  // Refresh account data
  const refreshAccount = useCallback(async () => {
    if (!account) return;

    try {
      const balances = await StellarService.getAccountBalance(account.publicKey);
      const transactions = await StellarService.getAccountTransactions(account.publicKey, 5);

      setAccount(prev => prev ? {
        ...prev,
        balances,
        transactions
      } : null);
    } catch (err: any) {
      console.error('Failed to refresh account:', err);
      setError('Failed to refresh account data');
    }
  }, [account]);

  // Switch network
  const switchNetwork = useCallback(async (network: StellarNetwork) => {
    try {
      StellarService.setNetwork(network);
      
      if (account) {
        // Refresh account data for new network
        await refreshAccount();
        setAccount(prev => prev ? { ...prev, network } : null);
      }
    } catch (err: any) {
      console.error('Failed to switch network:', err);
      setError('Failed to switch network');
    }
  }, [account, refreshAccount]);

  // Send payment
  const sendPayment = useCallback(async (params: {
    destinationKey: string;
    amount: string;
    assetCode?: string;
    memo?: string;
  }) => {
    if (!walletKit || !account) {
      throw new Error('Wallet not connected');
    }

    const { destinationKey, amount, memo } = params;

    try {
      // For demo purposes, we'll create a simple payment transaction XDR
      // In production, you'd build the transaction properly using StellarService
      const testAccount = await StellarService.createTestAccount();
      
      const result = await StellarService.sendPayment({
        sourceSecret: testAccount.secretKey,
        destinationKey,
        amount,
        memo
      });

      // Refresh account data
      await refreshAccount();
      
      return result;
    } catch (err: any) {
      console.error('Failed to send payment:', err);
      throw new Error(`Payment failed: ${err.message}`);
    }
  }, [walletKit, account, refreshAccount]);

  // Sign transaction
  const signTransaction = useCallback(async (xdr: string): Promise<string> => {
    if (!walletKit || !account) {
      throw new Error('Wallet not connected');
    }

    try {
      const { signedTxXdr } = await walletKit.signTransaction(xdr, {
        address: account.publicKey,
        networkPassphrase: StellarService.getNetworkPassphrase()
      });

      return signedTxXdr;
    } catch (err: any) {
      console.error('Failed to sign transaction:', err);
      throw new Error(`Transaction signing failed: ${err.message}`);
    }
  }, [walletKit, account]);

  // Create asset
  const createAsset = useCallback(async (params: {
    assetCode: string;
    limit?: string;
  }) => {
    if (!account) {
      throw new Error('Wallet not connected');
    }

    try {
      // Note: In production, this would require the issuer secret key
      // For demo purposes, we'll create a test account and use that
      const testAccount = await StellarService.createTestAccount();
      
      const asset = await StellarService.createAsset({
        issuerSecret: testAccount.secretKey,
        assetCode: params.assetCode,
        limit: params.limit
      });

      await refreshAccount();
      return asset;
    } catch (err: any) {
      console.error('Failed to create asset:', err);
      throw new Error(`Asset creation failed: ${err.message}`);
    }
  }, [account, refreshAccount]);

  // Tokenize asset
  const tokenizeAsset = useCallback(async (params: {
    assetType: 'real_estate' | 'commodity' | 'art' | 'bond';
    assetName: string;
    totalSupply: string;
    description: string;
    metadata: Record<string, any>;
  }) => {
    if (!account) {
      throw new Error('Wallet not connected');
    }

    try {
      // Note: In production, this would use the connected wallet's secret
      // For demo purposes, we'll create a test account
      const testAccount = await StellarService.createTestAccount();
      
      const result = await StellarService.tokenizeAsset({
        issuerSecret: testAccount.secretKey,
        ...params
      });

      await refreshAccount();
      return result;
    } catch (err: any) {
      console.error('Failed to tokenize asset:', err);
      throw new Error(`Asset tokenization failed: ${err.message}`);
    }
  }, [account, refreshAccount]);

  // Get portfolio value
  const getPortfolioValue = useCallback(async () => {
    if (!account) {
      throw new Error('Wallet not connected');
    }

    try {
      return await StellarService.getPortfolioValue(account.publicKey);
    } catch (err: any) {
      console.error('Failed to get portfolio value:', err);
      throw new Error(`Portfolio calculation failed: ${err.message}`);
    }
  }, [account]);

  // Utility methods
  const isValidAddress = useCallback((address: string): boolean => {
    return StellarService.isValidPublicKey(address);
  }, []);

  const formatAmount = useCallback((amount: string, decimals: number = 7): string => {
    return StellarService.formatAmount(amount, decimals);
  }, []);

  // Context value
  const contextValue: WalletContextState = {
    status,
    account,
    error,
    supportedWallets,
    connect,
    disconnect,
    refreshAccount,
    switchNetwork,
    sendPayment,
    signTransaction,
    createAsset,
    tokenizeAsset,
    getPortfolioValue,
    isValidAddress,
    formatAmount
  };

  return (
    <WalletContext.Provider value={contextValue}>
      {children}
    </WalletContext.Provider>
  );
}

// Hook to use wallet context
export function useWallet(): WalletContextState {
  const context = useContext(WalletContext);
  
  if (!context) {
    throw new Error('useWallet must be used within a WalletProvider');
  }
  
  return context;
} 