import { Metadata } from 'next';
import { DashboardStats } from '@/components/dashboard/dashboard-stats';
import { PortfolioOverview } from '@/components/dashboard/portfolio-overview';
import { AssetAllocation } from '@/components/dashboard/asset-allocation';
import { RecentTransactions } from '@/components/dashboard/recent-transactions';
import { MarketTrends } from '@/components/dashboard/market-trends';
import { QuickActions } from '@/components/dashboard/quick-actions';

export const metadata: Metadata = {
  title: 'Dashboard',
  description: 'Manage your tokenized assets and portfolio on StellarVault.',
};

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome back! Here's your portfolio overview.
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Stats */}
      <DashboardStats />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Portfolio & Allocation */}
        <div className="lg:col-span-2 space-y-6">
          <PortfolioOverview />
          <AssetAllocation />
        </div>

        {/* Right Column - Transactions & Trends */}
        <div className="space-y-6">
          <RecentTransactions />
          <MarketTrends />
        </div>
      </div>
    </div>
  );
} 