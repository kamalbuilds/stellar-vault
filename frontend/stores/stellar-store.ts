import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface StellarState {
  walletAddress: string | null;
  network: 'testnet' | 'mainnet';
  isConnected: boolean;
  balance: string | null;
  contractAddresses: {
    assetToken: string | null;
    compliance: string | null;
    portfolio: string | null;
    settlement: string | null;
  };
}

interface StellarActions {
  setWalletAddress: (address: string) => void;
  setNetwork: (network: 'testnet' | 'mainnet') => void;
  setConnected: (connected: boolean) => void;
  setBalance: (balance: string) => void;
  setContractAddress: (contract: keyof StellarState['contractAddresses'], address: string) => void;
  disconnect: () => void;
  reset: () => void;
}

type StellarStore = StellarState & StellarActions;

const initialState: StellarState = {
  walletAddress: null,
  network: 'testnet',
  isConnected: false,
  balance: null,
  contractAddresses: {
    assetToken: null,
    compliance: null,
    portfolio: null,
    settlement: null,
  },
};

export const useStellarStore = create<StellarStore>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Actions
      setWalletAddress: (walletAddress: string) => {
        set({ walletAddress });
      },

      setNetwork: (network: 'testnet' | 'mainnet') => {
        set({ network });
      },

      setConnected: (isConnected: boolean) => {
        set({ isConnected });
      },

      setBalance: (balance: string) => {
        set({ balance });
      },

      setContractAddress: (contract: keyof StellarState['contractAddresses'], address: string) => {
        set((state) => ({
          contractAddresses: {
            ...state.contractAddresses,
            [contract]: address,
          },
        }));
      },

      disconnect: () => {
        set({
          walletAddress: null,
          isConnected: false,
          balance: null,
        });
      },

      reset: () => {
        set(initialState);
      },
    }),
    {
      name: 'stellarvault-stellar',
      partialize: (state) => ({
        walletAddress: state.walletAddress,
        network: state.network,
        isConnected: state.isConnected,
        contractAddresses: state.contractAddresses,
      }),
    }
  )
); 