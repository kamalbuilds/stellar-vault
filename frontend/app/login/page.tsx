import { Metadata } from 'next';
import { LoginForm } from '@/components/auth/login-form';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Sign In',
  description: 'Sign in to your StellarVault account to manage your tokenized assets.',
};

export default function LoginPage() {
  return (
    <div className="min-h-screen flex">
      {/* Left side - Login form */}
      <div className="flex-1 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link href="/" className="flex items-center justify-center space-x-2 mb-8">
            <div className="w-10 h-10 gradient-stellar rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">SV</span>
            </div>
            <span className="font-bold text-2xl text-foreground">StellarVault</span>
          </Link>
          
          <h2 className="text-center text-3xl font-bold text-foreground">
            Sign in to your account
          </h2>
          <p className="mt-2 text-center text-sm text-muted-foreground">
            Or{' '}
            <Link
              href="/register"
              className="font-medium text-stellar-600 hover:text-stellar-500"
            >
              create a new account
            </Link>
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-card py-8 px-4 shadow-lg sm:rounded-lg sm:px-10 border border-border">
            <LoginForm />
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground">
            Protected by enterprise-grade security
          </p>
        </div>
      </div>

      {/* Right side - Hero image/content */}
      <div className="hidden lg:block relative flex-1 bg-gradient-to-br from-stellar-600 to-vault-600">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative h-full flex flex-col justify-center px-12 text-white">
          <div className="max-w-md">
            <h3 className="text-3xl font-bold mb-4">
              Welcome back to the future of finance
            </h3>
            <p className="text-lg text-white/90 mb-8">
              Access your tokenized real-world assets, manage your portfolio, 
              and trade on the global marketplace.
            </p>
            
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>$1B+ in assets under management</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>500+ institutional clients</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>99.9% uptime guarantee</span>
              </div>
            </div>
          </div>
          
          {/* Floating elements */}
          <div className="absolute top-20 right-20 w-32 h-32 bg-white/10 rounded-full blur-xl"></div>
          <div className="absolute bottom-20 left-20 w-24 h-24 bg-white/10 rounded-full blur-xl"></div>
        </div>
      </div>
    </div>
  );
} 