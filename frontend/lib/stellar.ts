// stellar-vault/frontend/lib/stellar.ts
import * as StellarSdk from '@stellar/stellar-sdk'

// Network configurations
export enum StellarNetwork {
  TESTNET = 'testnet',
  FUTURENET = 'futurenet',
  MAINNET = 'mainnet'
}

export interface NetworkConfig {
  horizonUrl: string;
  rpcUrl: string;
  passphrase: string;
  friendbotUrl?: string;
}

export interface AssetInfo {
  code: string;
  issuer?: string;
  type: 'native' | 'credit_alphanum4' | 'credit_alphanum12' | 'contract';
  contractId?: string;
  balance: string;
  limit?: string;
}

export interface TransactionInfo {
  id: string;
  type: string;
  amount?: string;
  asset?: string;
  timestamp: string;
  from: string;
  to?: string;
  memo?: string;
  successful: boolean;
}

export interface TokenizationParams {
  assetType: 'real_estate' | 'commodity' | 'art' | 'bond';
  assetName: string;
  totalSupply: string;
  description: string;
  metadata: Record<string, any>;
  complianceRules?: string[];
}

class StellarServiceClass {
  private currentNetwork: StellarNetwork = StellarNetwork.TESTNET;
  private horizonServer!: StellarSdk.Horizon.Server;
  private rpcServer!: StellarSdk.SorobanRpc.Server;
  private networkConfig!: NetworkConfig;

  constructor() {
    this.setNetwork(StellarNetwork.TESTNET);
  }

  private getNetworkConfig(network: StellarNetwork): NetworkConfig {
    switch (network) {
      case StellarNetwork.TESTNET:
        return {
          horizonUrl: 'https://horizon-testnet.stellar.org',
          rpcUrl: 'https://soroban-testnet.stellar.org',
          passphrase: StellarSdk.Networks.TESTNET,
          friendbotUrl: 'https://friendbot.stellar.org'
        };
      case StellarNetwork.FUTURENET:
        return {
          horizonUrl: 'https://horizon-futurenet.stellar.org',
          rpcUrl: 'https://rpc-futurenet.stellar.org',
          passphrase: StellarSdk.Networks.FUTURENET,
          friendbotUrl: 'https://friendbot-futurenet.stellar.org'
        };
      case StellarNetwork.MAINNET:
        return {
          horizonUrl: 'https://horizon.stellar.org',
          rpcUrl: 'https://soroban-rpc.stellar.org',
          passphrase: StellarSdk.Networks.PUBLIC
        };
      default:
        throw new Error(`Unsupported network: ${network}`);
    }
  }

  setNetwork(network: StellarNetwork): void {
    this.currentNetwork = network;
    this.networkConfig = this.getNetworkConfig(network);
    this.horizonServer = new StellarSdk.Horizon.Server(this.networkConfig.horizonUrl);
    this.rpcServer = new StellarSdk.SorobanRpc.Server(this.networkConfig.rpcUrl);
  }

  getCurrentNetwork(): StellarNetwork {
    return this.currentNetwork;
  }

  getNetworkPassphrase(): string {
    return this.networkConfig.passphrase;
  }

  // Account Management
  async createTestAccount(): Promise<{ publicKey: string; secretKey: string }> {
    if (this.currentNetwork === StellarNetwork.MAINNET) {
      throw new Error('Cannot create test accounts on mainnet');
    }

    const pair = StellarSdk.Keypair.random();
    
    try {
      // Fund the account using friendbot (testnet/futurenet only)
      const response = await fetch(
        `${this.networkConfig.friendbotUrl}?addr=${pair.publicKey()}`
      );
      
      if (!response.ok) {
        throw new Error(`Friendbot request failed: ${response.statusText}`);
      }
      
      return {
        publicKey: pair.publicKey(),
        secretKey: pair.secret(),
      };
    } catch (error: any) {
      console.error('Error creating test account:', error);
      throw new Error(`Failed to create test account: ${error.message}`);
    }
  }

  async getAccountInfo(publicKey: string): Promise<any> {
    try {
      const account = await this.horizonServer.loadAccount(publicKey);
      return {
        id: account.id,
        sequence: account.sequence,
        balances: this.parseBalances(account.balances),
        signers: account.signers,
        thresholds: account.thresholds,
        flags: account.flags,
        data: account.data
      };
    } catch (error: any) {
      if (error.response && error.response.status === 404) {
        throw new Error('Account not found on the network');
      }
      console.error('Error getting account info:', error);
      throw new Error(`Failed to get account info: ${error.message}`);
    }
  }

  async getAccountBalance(publicKey: string): Promise<AssetInfo[]> {
    try {
      const account = await this.horizonServer.loadAccount(publicKey);
      return this.parseBalances(account.balances);
    } catch (error: any) {
      console.error('Error getting balance:', error);
      throw new Error(`Failed to get account balance: ${error.message}`);
    }
  }

  private parseBalances(balances: any[]): AssetInfo[] {
    return balances.map((balance: any) => ({
      code: balance.asset_type === 'native' ? 'XLM' : balance.asset_code,
      issuer: balance.asset_issuer,
      type: balance.asset_type,
      balance: balance.balance,
      limit: balance.limit
    }));
  }

  // Transaction Management
  async sendPayment(params: {
    sourceSecret: string;
    destinationKey: string;
    amount: string;
    asset?: StellarSdk.Asset;
    memo?: string;
  }): Promise<any> {
    const { sourceSecret, destinationKey, amount, asset = StellarSdk.Asset.native(), memo } = params;
    
    try {
      const sourceKeypair = StellarSdk.Keypair.fromSecret(sourceSecret);
      const sourceAccount = await this.horizonServer.loadAccount(sourceKeypair.publicKey());
      
      const transactionBuilder = new StellarSdk.TransactionBuilder(sourceAccount, {
        fee: StellarSdk.BASE_FEE,
        networkPassphrase: this.networkConfig.passphrase,
      })
        .addOperation(
          StellarSdk.Operation.payment({
            destination: destinationKey,
            asset: asset,
            amount: amount,
          })
        )
        .setTimeout(180);

      if (memo) {
        transactionBuilder.addMemo(StellarSdk.Memo.text(memo));
      }

      const transaction = transactionBuilder.build();
      transaction.sign(sourceKeypair);
      
      const result = await this.horizonServer.submitTransaction(transaction);
      return result;
    } catch (error: any) {
      console.error('Error sending payment:', error);
      throw new Error(`Failed to send payment: ${error.message}`);
    }
  }

  async getAccountTransactions(publicKey: string, limit: number = 10): Promise<TransactionInfo[]> {
    try {
      const transactions = await this.horizonServer
        .transactions()
        .forAccount(publicKey)
        .order('desc')
        .limit(limit)
        .call();

      const transactionInfos: TransactionInfo[] = [];
      
      for (const tx of transactions.records) {
        const operations = await tx.operations();
        for (const op of operations.records) {
          transactionInfos.push({
            id: tx.id,
            type: op.type,
            amount: (op as any).amount,
            asset: (op as any).asset_type === 'native' ? 'XLM' : (op as any).asset_code,
            timestamp: tx.created_at,
            from: (op as any).from || tx.source_account,
            to: (op as any).to,
            memo: tx.memo,
            successful: tx.successful
          });
        }
      }

      return transactionInfos;
    } catch (error: any) {
      console.error('Error fetching transactions:', error);
      throw new Error(`Failed to fetch transactions: ${error.message}`);
    }
  }

  // Asset Management
  async createAsset(params: {
    issuerSecret: string;
    assetCode: string;
    limit?: string;
  }): Promise<StellarSdk.Asset> {
    const { issuerSecret, assetCode, limit } = params;
    
    try {
      const issuerKeypair = StellarSdk.Keypair.fromSecret(issuerSecret);
      const asset = new StellarSdk.Asset(assetCode, issuerKeypair.publicKey());
      
      // If limit is specified, we need to create a distribution account and set up trustline
      if (limit) {
        const distributorKeypair = StellarSdk.Keypair.random();
        
        // Fund distributor account
        if (this.networkConfig.friendbotUrl) {
          await fetch(`${this.networkConfig.friendbotUrl}?addr=${distributorKeypair.publicKey()}`);
        }
        
        const distributorAccount = await this.horizonServer.loadAccount(distributorKeypair.publicKey());
        
        // Create trustline from distributor to asset
        const trustlineTransaction = new StellarSdk.TransactionBuilder(distributorAccount, {
          fee: StellarSdk.BASE_FEE,
          networkPassphrase: this.networkConfig.passphrase,
        })
          .addOperation(
            StellarSdk.Operation.changeTrust({
              asset: asset,
              limit: limit
            })
          )
          .setTimeout(180)
          .build();

        trustlineTransaction.sign(distributorKeypair);
        await this.horizonServer.submitTransaction(trustlineTransaction);
      }
      
      return asset;
    } catch (error: any) {
      console.error('Error creating asset:', error);
      throw new Error(`Failed to create asset: ${error.message}`);
    }
  }

  async getSupportedAssets(): Promise<AssetInfo[]> {
    try {
      const assets = await this.horizonServer.assets().call();
      return assets.records.map((asset: any) => ({
        code: asset.asset_code || 'XLM',
        issuer: asset.asset_issuer,
        type: asset.asset_type,
        balance: '0' // Will be updated when fetching account-specific balances
      }));
    } catch (error: any) {
      console.error('Error fetching assets:', error);
      throw new Error(`Failed to fetch supported assets: ${error.message}`);
    }
  }

  // Asset Tokenization (Integration with Soroban contracts)
  async tokenizeAsset(params: TokenizationParams & { 
    issuerSecret: string;
    contractId?: string;
  }): Promise<{
    asset: StellarSdk.Asset;
    contractId?: string;
    transactionResult: any;
  }> {
    const { issuerSecret, assetType, assetName, totalSupply, description, metadata, contractId } = params;
    
    try {
      const issuerKeypair = StellarSdk.Keypair.fromSecret(issuerSecret);
      
      if (contractId) {
        // Use Soroban contract for advanced tokenization
        const contract = new StellarSdk.Contract(contractId);
        const sourceAccount = await this.rpcServer.getAccount(issuerKeypair.publicKey());
        
        const transaction = new StellarSdk.TransactionBuilder(sourceAccount, {
          fee: StellarSdk.BASE_FEE,
          networkPassphrase: this.networkConfig.passphrase,
        })
          .addOperation(
            contract.call(
              'tokenize_asset',
              StellarSdk.nativeToScVal(assetType, { type: 'string' }),
              StellarSdk.nativeToScVal(assetName, { type: 'string' }),
              StellarSdk.nativeToScVal(totalSupply, { type: 'string' }),
              StellarSdk.nativeToScVal(description, { type: 'string' }),
              StellarSdk.nativeToScVal(metadata, { type: 'map' })
            )
          )
          .setTimeout(180)
          .build();

        transaction.sign(issuerKeypair);
        const result = await this.rpcServer.sendTransaction(transaction);
        
        return {
          asset: new StellarSdk.Asset(assetName, issuerKeypair.publicKey()),
          contractId,
          transactionResult: result
        };
      } else {
        // Use traditional Stellar asset
        const asset = await this.createAsset({
          issuerSecret,
          assetCode: assetName,
          limit: totalSupply
        });
        
        return {
          asset,
          transactionResult: { success: true }
        };
      }
    } catch (error: any) {
      console.error('Error tokenizing asset:', error);
      throw new Error(`Failed to tokenize asset: ${error.message}`);
    }
  }

  // Compliance and KYC Integration
  async checkCompliance(publicKey: string, assetCode: string): Promise<{
    compliant: boolean;
    restrictions: string[];
    kycStatus: 'verified' | 'pending' | 'rejected' | 'not_required';
  }> {
    try {
      // This would integrate with your AI-powered compliance system
      // For now, returning a basic check
      const account = await this.getAccountInfo(publicKey);
      
      return {
        compliant: true,
        restrictions: [],
        kycStatus: 'not_required' // Would be determined by your compliance contract
      };
    } catch (error: any) {
      console.error('Error checking compliance:', error);
      throw new Error(`Failed to check compliance: ${error.message}`);
    }
  }

  // Portfolio Management
  async getPortfolioValue(publicKey: string): Promise<{
    totalValue: string;
    assets: Array<AssetInfo & { value: string; percentage: number }>;
  }> {
    try {
      const balances = await this.getAccountBalance(publicKey);
      
      // This would integrate with your AI valuation system
      // For now, returning basic calculation
      let totalValue = 0;
      const assetsWithValue = balances.map(asset => {
        const value = parseFloat(asset.balance) * 1; // Would get real price from AI engine
        totalValue += value;
        return {
          ...asset,
          value: value.toString(),
          percentage: 0 // Will be calculated after total
        };
      });

      // Calculate percentages
      assetsWithValue.forEach(asset => {
        asset.percentage = totalValue > 0 ? (parseFloat(asset.value) / totalValue) * 100 : 0;
      });

      return {
        totalValue: totalValue.toString(),
        assets: assetsWithValue
      };
    } catch (error: any) {
      console.error('Error getting portfolio value:', error);
      throw new Error(`Failed to get portfolio value: ${error.message}`);
    }
  }

  // Cross-border Settlement
  async initiateCrossBorderPayment(params: {
    sourceSecret: string;
    destinationKey: string;
    amount: string;
    asset: StellarSdk.Asset;
    settlementCurrency: string;
    complianceData: Record<string, any>;
  }): Promise<any> {
    const { sourceSecret, destinationKey, amount, asset, settlementCurrency, complianceData } = params;
    
    try {
      // This would integrate with your cross-border settlement contract
      const sourceKeypair = StellarSdk.Keypair.fromSecret(sourceSecret);
      const sourceAccount = await this.horizonServer.loadAccount(sourceKeypair.publicKey());
      
      const transaction = new StellarSdk.TransactionBuilder(sourceAccount, {
        fee: StellarSdk.BASE_FEE,
        networkPassphrase: this.networkConfig.passphrase,
      })
        .addOperation(
          StellarSdk.Operation.payment({
            destination: destinationKey,
            asset: asset,
            amount: amount,
          })
        )
        .addMemo(StellarSdk.Memo.text(JSON.stringify({
          settlementCurrency,
          complianceData
        })))
        .setTimeout(180)
        .build();

      transaction.sign(sourceKeypair);
      return await this.horizonServer.submitTransaction(transaction);
    } catch (error: any) {
      console.error('Error initiating cross-border payment:', error);
      throw new Error(`Failed to initiate cross-border payment: ${error.message}`);
    }
  }

  // Utility Methods
  isValidPublicKey(publicKey: string): boolean {
    try {
      StellarSdk.Keypair.fromPublicKey(publicKey);
      return true;
    } catch {
      return false;
    }
  }

  isValidSecret(secret: string): boolean {
    try {
      StellarSdk.Keypair.fromSecret(secret);
      return true;
    } catch {
      return false;
    }
  }

  formatAmount(amount: string, decimals: number = 7): string {
    return parseFloat(amount).toFixed(decimals);
  }

  async estimateTransactionFee(operations: StellarSdk.Operation[]): Promise<string> {
    try {
      const fee = Number(StellarSdk.BASE_FEE) * operations.length;
      return fee.toString();
    } catch (error: any) {
      console.error('Error estimating fee:', error);
      return StellarSdk.BASE_FEE.toString();
    }
  }
}

// Export singleton instance
export const StellarService = new StellarServiceClass();