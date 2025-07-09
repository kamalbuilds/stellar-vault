import Link from 'next/link';
import { ArrowRight, Sparkles } from 'lucide-react';

export function CTA() {
  return (
    <section className="py-16 lg:py-24 bg-gradient-to-r from-stellar-600 via-stellar-700 to-vault-600 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 bg-black/20"></div>
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-white/5 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-white/5 rounded-full blur-3xl"></div>

      <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="inline-flex items-center px-4 py-2 bg-white/10 rounded-full text-white/90 text-sm font-medium mb-8">
          <Sparkles className="w-4 h-4 mr-2" />
          Join the RWA Revolution
        </div>

        <h2 className="text-3xl lg:text-5xl font-bold text-white mb-6">
          Ready to Tokenize Your Assets?
        </h2>

        <p className="text-xl text-white/90 mb-8 max-w-2xl mx-auto">
          Join thousands of investors and institutions already using StellarVault to unlock 
          the value of real-world assets through blockchain technology.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-6 mb-12">
          <Link
            href="/register"
            className="group inline-flex items-center px-8 py-4 bg-white text-stellar-700 font-semibold rounded-lg hover:bg-gray-100 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            Get Started Free
            <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          
          <Link
            href="/contact"
            className="inline-flex items-center px-8 py-4 border-2 border-white/30 text-white font-semibold rounded-lg hover:bg-white/10 transition-colors"
          >
            Schedule Demo
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-3xl mx-auto">
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">$0</div>
            <div className="text-white/80 text-sm">Setup Cost</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">5 min</div>
            <div className="text-white/80 text-sm">To Get Started</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white mb-1">24/7</div>
            <div className="text-white/80 text-sm">Expert Support</div>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-white/20">
          <p className="text-white/70 text-sm">
            Trusted by 500+ institutions • $1B+ assets tokenized • Available globally
          </p>
        </div>
      </div>
    </section>
  );
} 