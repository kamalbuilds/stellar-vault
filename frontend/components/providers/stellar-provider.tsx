'use client';

import { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { 
  Horizon, 
  Networks, 
  SorobanRpc,
  xdr 
} from '@stellar/stellar-sdk';
import { useStellarStore } from '@/stores/stellar-store';

interface StellarContextType {
  server: Horizon.Server | null;
  sorobanServer: SorobanRpc.Server | null;
  network: string;
  isConnected: boolean;
  walletAddress: string | null;
  connectWallet: () => Promise<void>;
  disconnectWallet: () => void;
  signTransaction: (transaction: any) => Promise<any>;
  submitTransaction: (transaction: any) => Promise<any>;
}

const StellarContext = createContext<StellarContextType | undefined>(undefined);

interface StellarProviderProps {
  children: ReactNode;
}

export function StellarProvider({ children }: StellarProviderProps) {
  const {
    walletAddress,
    network,
    isConnected,
    setWalletAddress,
    setNetwork,
    setConnected,
    disconnect
  } = useStellarStore();

  const [server, setServer] = useState<Horizon.Server | null>(null);
  const [sorobanServer, setSorobanServer] = useState<SorobanRpc.Server | null>(null);

  // Initialize Stellar servers
  useEffect(() => {
    const horizonUrl = network === 'mainnet' 
      ? 'https://horizon.stellar.org'
      : 'https://horizon-testnet.stellar.org';
    
    const sorobanUrl = network === 'mainnet'
      ? 'https://soroban-rpc.stellar.org'
      : 'https://soroban-rpc.stellar.org:443';

    setServer(new Horizon.Server(horizonUrl));
    setSorobanServer(new SorobanRpc.Server(sorobanUrl));
  }, [network]);

  // Check for existing wallet connection
  useEffect(() => {
    const checkConnection = async () => {
      if (typeof window !== 'undefined' && window.freighter) {
        try {
          const isAllowed = await window.freighter.isAllowed();
          if (isAllowed) {
            const publicKey = await window.freighter.getPublicKey();
            setWalletAddress(publicKey);
            setConnected(true);
          }
        } catch (error) {
          console.error('Error checking wallet connection:', error);
        }
      }
    };

    checkConnection();
  }, [setWalletAddress, setConnected]);

  const connectWallet = async () => {
    try {
      if (!window.freighter) {
        throw new Error('Freighter wallet not installed');
      }

      await window.freighter.requestAccess();
      const publicKey = await window.freighter.getPublicKey();
      
      setWalletAddress(publicKey);
      setConnected(true);
    } catch (error) {
      console.error('Error connecting wallet:', error);
      throw error;
    }
  };

  const disconnectWallet = () => {
    disconnect();
  };

  const signTransaction = async (transaction: any) => {
    try {
      if (!window.freighter || !isConnected) {
        throw new Error('Wallet not connected');
      }

      const networkPassphrase = network === 'mainnet' 
        ? Networks.PUBLIC 
        : Networks.TESTNET;

      const signedTransaction = await window.freighter.signTransaction(
        transaction,
        {
          networkPassphrase,
          accountToSign: walletAddress
        }
      );

      return signedTransaction;
    } catch (error) {
      console.error('Error signing transaction:', error);
      throw error;
    }
  };

  const submitTransaction = async (signedTransaction: any) => {
    try {
      if (!server) {
        throw new Error('Stellar server not initialized');
      }

      const result = await server.submitTransaction(signedTransaction);
      return result;
    } catch (error) {
      console.error('Error submitting transaction:', error);
      throw error;
    }
  };

  const value: StellarContextType = {
    server,
    sorobanServer,
    network,
    isConnected,
    walletAddress,
    connectWallet,
    disconnectWallet,
    signTransaction,
    submitTransaction,
  };

  return (
    <StellarContext.Provider value={value}>
      {children}
    </StellarContext.Provider>
  );
}

export function useStellar() {
  const context = useContext(StellarContext);
  if (context === undefined) {
    throw new Error('useStellar must be used within a StellarProvider');
  }
  return context;
}

// Extend window object for Freighter wallet
declare global {
  interface Window {
    freighter?: {
      isConnected: () => Promise<boolean>;
      isAllowed: () => Promise<boolean>;
      requestAccess: () => Promise<void>;
      getPublicKey: () => Promise<string>;
      signTransaction: (
        transaction: string,
        options?: {
          networkPassphrase?: string;
          accountToSign?: string;
        }
      ) => Promise<string>;
    };
  }
} 