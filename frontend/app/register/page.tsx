import { Metadata } from 'next';
import { RegisterForm } from '@/components/auth/register-form';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Create Account',
  description: 'Create your StellarVault account to start tokenizing real-world assets.',
};

export default function RegisterPage() {
  return (
    <div className="min-h-screen flex">
      {/* Left side - Register form */}
      <div className="flex-1 flex flex-col justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link href="/" className="flex items-center justify-center space-x-2 mb-8">
            <div className="w-10 h-10 gradient-stellar rounded-lg flex items-center justify-center">
              <span className="text-white font-bold">SV</span>
            </div>
            <span className="font-bold text-2xl text-foreground">StellarVault</span>
          </Link>
          
          <h2 className="text-center text-3xl font-bold text-foreground">
            Create your account
          </h2>
          <p className="mt-2 text-center text-sm text-muted-foreground">
            Already have an account?{' '}
            <Link
              href="/login"
              className="font-medium text-stellar-600 hover:text-stellar-500"
            >
              Sign in
            </Link>
          </p>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-card py-8 px-4 shadow-lg sm:rounded-lg sm:px-10 border border-border">
            <RegisterForm />
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground">
            By creating an account, you agree to our{' '}
            <Link href="/terms" className="text-stellar-600 hover:text-stellar-500">
              Terms of Service
            </Link>{' '}
            and{' '}
            <Link href="/privacy" className="text-stellar-600 hover:text-stellar-500">
              Privacy Policy
            </Link>
          </p>
        </div>
      </div>

      {/* Right side - Hero content */}
      <div className="hidden lg:block relative flex-1 bg-gradient-to-br from-vault-600 to-stellar-600">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative h-full flex flex-col justify-center px-12 text-white">
          <div className="max-w-md">
            <h3 className="text-3xl font-bold mb-4">
              Join the RWA Revolution
            </h3>
            <p className="text-lg text-white/90 mb-8">
              Tokenize real-world assets, access global markets, and build your 
              diversified portfolio with AI-powered insights.
            </p>
            
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>$16T market opportunity</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>95%+ AI valuation accuracy</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>SEC compliant platform</span>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-2 h-2 bg-white rounded-full"></div>
                <span>Global 24/7 trading</span>
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