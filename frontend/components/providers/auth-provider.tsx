'use client';

import { createContext, useContext, useEffect } from 'react';
import { useAuthStore } from '@/stores/auth-store';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

interface AuthContextType {
  user: any;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (token: string) => void;
  logout: () => void;
  refreshUser: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const {
    user,
    token,
    isAuthenticated,
    setUser,
    setToken,
    clearAuth,
  } = useAuthStore();

  // Fetch user data when token exists
  const { data: userData, isLoading, refetch } = useQuery({
    queryKey: ['user', token],
    queryFn: () => api.auth.getProfile(),
    enabled: !!token && !user,
    retry: false,
  });

  useEffect(() => {
    if (userData && !user) {
      setUser(userData);
    }
  }, [userData, user, setUser]);

  // Check for stored token on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('stellarvault_token');
    if (storedToken && !token) {
      setToken(storedToken);
    }
  }, [token, setToken]);

  const login = (newToken: string) => {
    localStorage.setItem('stellarvault_token', newToken);
    setToken(newToken);
  };

  const logout = () => {
    localStorage.removeItem('stellarvault_token');
    clearAuth();
  };

  const refreshUser = () => {
    refetch();
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated,
    login,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
} 