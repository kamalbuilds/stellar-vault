import Link from 'next/link';
import { ArrowRight, Shield, TrendingUp, Zap, Globe } from 'lucide-react';

export function Hero() {
  return (
    <section className="relative overflow-hidden bg-background">
      {/* Background gradient */}
      <div className="absolute inset-0 gradient-mesh opacity-50"></div>
      
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-stellar-100 text-stellar-800 dark:bg-stellar-900/20 dark:text-stellar-300 text-sm font-medium mb-6">
            <Zap className="w-4 h-4 mr-2" />
            Now live on Stellar Testnet
          </div>

          {/* Main heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground mb-6">
            Unlock{' '}
            <span className="bg-gradient-to-r from-stellar-600 to-vault-600 bg-clip-text text-transparent">
              $16 Trillion
            </span>
            {' '}in Real-World Assets
          </h1>

          {/* Subheading */}
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto mb-8 leading-relaxed">
            AI-powered platform for tokenizing real-world assets on Stellar blockchain. 
            Make illiquid assets accessible through fractional ownership and intelligent valuation.
          </p>

          {/* Feature highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto mb-12">
            <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card border border-border">
              <Shield className="w-6 h-6 text-stellar-600" />
              <span className="font-medium">SEC Compliant</span>
            </div>
            <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card border border-border">
              <TrendingUp className="w-6 h-6 text-vault-600" />
              <span className="font-medium">AI Valuation</span>
            </div>
            <div className="flex items-center justify-center space-x-3 p-4 rounded-lg bg-card border border-border">
              <Globe className="w-6 h-6 text-success-600" />
              <span className="font-medium">Global Access</span>
            </div>
          </div>

          {/* CTA buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6">
            <Link
              href="/register"
              className="group inline-flex items-center px-8 py-4 bg-stellar-600 text-white font-semibold rounded-lg hover:bg-stellar-700 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              Start Tokenizing
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            
            <Link
              href="/demo"
              className="inline-flex items-center px-8 py-4 border border-border text-foreground font-semibold rounded-lg hover:bg-muted transition-colors"
            >
              View Demo
            </Link>
          </div>

          {/* Trust indicators */}
          <div className="mt-16 pt-8 border-t border-border">
            <p className="text-sm text-muted-foreground mb-6">Trusted by institutions and backed by</p>
            <div className="flex items-center justify-center space-x-8 opacity-60">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-stellar-600 rounded-lg"></div>
                <span className="font-semibold">Stellar</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gray-600 rounded-lg"></div>
                <span className="font-semibold">DraperU</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-blue-600 rounded-lg"></div>
                <span className="font-semibold">Fireblocks</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Floating elements */}
      <div className="absolute top-1/4 left-10 w-24 h-24 bg-stellar-500/10 rounded-full blur-xl"></div>
      <div className="absolute bottom-1/4 right-10 w-32 h-32 bg-vault-500/10 rounded-full blur-xl"></div>
    </section>
  );
} 