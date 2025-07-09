import { useAuthStore } from '@/stores/auth-store';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

interface ApiError {
  message: string;
  status: number;
  errors?: Record<string, string[]>;
}

class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const { token } = useAuthStore.getState();
    
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw {
          message: errorData.message || 'An error occurred',
          status: response.status,
          errors: errorData.errors,
        } as ApiError;
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return response.json();
      }
      
      return response.text() as any;
    } catch (error) {
      if (error instanceof TypeError) {
        throw {
          message: 'Network error - please check your connection',
          status: 0,
        } as ApiError;
      }
      throw error;
    }
  }

  private get<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET', ...options });
  }

  private post<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });
  }

  private put<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
      ...options,
    });
  }

  private delete<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE', ...options });
  }

  // Authentication endpoints
  auth = {
    login: (credentials: { email: string; password: string }) =>
      this.post<{ token: string; user: any }>('/auth/login', credentials),
    
    register: (userData: {
      email: string;
      password: string;
      firstName: string;
      lastName: string;
      role?: string;
    }) => this.post<{ token: string; user: any }>('/auth/register', userData),
    
    logout: () => this.post('/auth/logout'),
    
    getProfile: () => this.get<any>('/auth/profile'),
    
    updateProfile: (data: Partial<any>) =>
      this.put<any>('/auth/profile', data),
    
    forgotPassword: (email: string) =>
      this.post('/auth/forgot-password', { email }),
    
    resetPassword: (token: string, password: string) =>
      this.post('/auth/reset-password', { token, password }),
  };

  // Asset management endpoints
  assets = {
    getAll: (params?: {
      page?: number;
      limit?: number;
      type?: string;
      status?: string;
    }) => this.get<any>('/assets', { 
      headers: params ? { 'X-Query-Params': JSON.stringify(params) } : {}
    }),
    
    getById: (id: string) => this.get<any>(`/assets/${id}`),
    
    create: (assetData: {
      name: string;
      type: string;
      value: number;
      description: string;
      metadata: Record<string, any>;
    }) => this.post<any>('/assets', assetData),
    
    update: (id: string, data: Partial<any>) =>
      this.put<any>(`/assets/${id}`, data),
    
    delete: (id: string) => this.delete(`/assets/${id}`),
    
    tokenize: (id: string, params: {
      totalTokens: number;
      tokenPrice: number;
    }) => this.post<any>(`/assets/${id}/tokenize`, params),
    
    getValuation: (id: string) => this.get<any>(`/assets/${id}/valuation`),
  };

  // Portfolio endpoints
  portfolio = {
    get: () => this.get<any>('/portfolio'),
    
    getPerformance: (period?: string) =>
      this.get<any>(`/portfolio/performance${period ? `?period=${period}` : ''}`),
    
    rebalance: () => this.post<any>('/portfolio/rebalance'),
    
    getRecommendations: () => this.get<any>('/portfolio/recommendations'),
  };

  // Trading endpoints
  trading = {
    getOrders: (params?: {
      status?: string;
      type?: string;
      page?: number;
      limit?: number;
    }) => this.get<any>('/trading/orders', {
      headers: params ? { 'X-Query-Params': JSON.stringify(params) } : {}
    }),
    
    createOrder: (orderData: {
      assetId: string;
      type: 'buy' | 'sell';
      quantity: number;
      price?: number;
    }) => this.post<any>('/trading/orders', orderData),
    
    cancelOrder: (orderId: string) =>
      this.delete(`/trading/orders/${orderId}`),
    
    getMarketData: (assetId: string) =>
      this.get<any>(`/trading/market-data/${assetId}`),
  };

  // Compliance endpoints
  compliance = {
    getKycStatus: () => this.get<any>('/compliance/kyc/status'),
    
    submitKyc: (kycData: FormData) =>
      this.post<any>('/compliance/kyc/submit', kycData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      }),
    
    getTransactionReports: (params?: {
      startDate?: string;
      endDate?: string;
      type?: string;
    }) => this.get<any>('/compliance/reports/transactions', {
      headers: params ? { 'X-Query-Params': JSON.stringify(params) } : {}
    }),
  };

  // Analytics endpoints
  analytics = {
    getDashboard: () => this.get<any>('/analytics/dashboard'),
    
    getMarketOverview: () => this.get<any>('/analytics/market-overview'),
    
    getAssetPerformance: (assetId: string, period?: string) =>
      this.get<any>(`/analytics/assets/${assetId}/performance${period ? `?period=${period}` : ''}`),
    
    getRiskMetrics: () => this.get<any>('/analytics/risk-metrics'),
  };

  // Stellar blockchain endpoints
  stellar = {
    getBalance: (address: string) =>
      this.get<any>(`/stellar/balance/${address}`),
    
    getTransactions: (address: string, params?: {
      limit?: number;
      cursor?: string;
    }) => this.get<any>(`/stellar/transactions/${address}`, {
      headers: params ? { 'X-Query-Params': JSON.stringify(params) } : {}
    }),
    
    submitTransaction: (transactionXdr: string) =>
      this.post<any>('/stellar/submit-transaction', { transactionXdr }),
    
    getContractData: (contractId: string, key: string) =>
      this.get<any>(`/stellar/contracts/${contractId}/data/${key}`),
  };
}

export const api = new ApiClient(API_BASE_URL);
export type { ApiError }; 