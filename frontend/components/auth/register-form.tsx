'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Eye, EyeOff, Loader2, User, Building } from 'lucide-react';
import { api } from '@/lib/api';
import toast from 'react-hot-toast';

const registerSchema = z.object({
  firstName: z.string().min(2, 'First name must be at least 2 characters'),
  lastName: z.string().min(2, 'Last name must be at least 2 characters'),
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters'),
  confirmPassword: z.string(),
  accountType: z.enum(['individual', 'institution']),
  companyName: z.string().optional(),
  agreeToTerms: z.boolean().refine((val) => val === true, 'You must agree to the terms'),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

type RegisterFormData = z.infer<typeof registerSchema>;

export function RegisterForm() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [accountType, setAccountType] = useState<'individual' | 'institution'>('individual');
  const router = useRouter();

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      accountType: 'individual',
    },
  });

  const watchAccountType = watch('accountType');

  const onSubmit = async (data: RegisterFormData) => {
    setIsLoading(true);
    try {
      await api.auth.register(data);
      toast.success('Account created successfully! Please check your email to verify your account.');
      router.push('/login');
    } catch (error: any) {
      toast.error(error.message || 'Failed to create account');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAccountTypeChange = (type: 'individual' | 'institution') => {
    setAccountType(type);
    setValue('accountType', type);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {/* Account Type Selection */}
      <div>
        <label className="block text-sm font-medium text-foreground mb-3">
          Account Type
        </label>
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => handleAccountTypeChange('individual')}
            className={`p-3 rounded-lg border-2 transition-all ${
              watchAccountType === 'individual'
                ? 'border-stellar-500 bg-stellar-50 text-stellar-700 dark:bg-stellar-900/20 dark:text-stellar-300'
                : 'border-border bg-card text-muted-foreground hover:bg-muted'
            }`}
          >
            <User className="w-5 h-5 mx-auto mb-1" />
            <div className="text-sm font-medium">Individual</div>
          </button>
          <button
            type="button"
            onClick={() => handleAccountTypeChange('institution')}
            className={`p-3 rounded-lg border-2 transition-all ${
              watchAccountType === 'institution'
                ? 'border-stellar-500 bg-stellar-50 text-stellar-700 dark:bg-stellar-900/20 dark:text-stellar-300'
                : 'border-border bg-card text-muted-foreground hover:bg-muted'
            }`}
          >
            <Building className="w-5 h-5 mx-auto mb-1" />
            <div className="text-sm font-medium">Institution</div>
          </button>
        </div>
      </div>

      {/* Company Name (for institutions) */}
      {watchAccountType === 'institution' && (
        <div>
          <label htmlFor="companyName" className="block text-sm font-medium text-foreground">
            Company Name
          </label>
          <div className="mt-1">
            <input
              id="companyName"
              type="text"
              className="appearance-none block w-full px-3 py-2 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
              placeholder="Enter your company name"
              {...register('companyName')}
            />
            {errors.companyName && (
              <p className="mt-1 text-sm text-error-600">{errors.companyName.message}</p>
            )}
          </div>
        </div>
      )}

      {/* Name Fields */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label htmlFor="firstName" className="block text-sm font-medium text-foreground">
            First Name
          </label>
          <div className="mt-1">
            <input
              id="firstName"
              type="text"
              autoComplete="given-name"
              className="appearance-none block w-full px-3 py-2 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
              placeholder="John"
              {...register('firstName')}
            />
            {errors.firstName && (
              <p className="mt-1 text-sm text-error-600">{errors.firstName.message}</p>
            )}
          </div>
        </div>
        <div>
          <label htmlFor="lastName" className="block text-sm font-medium text-foreground">
            Last Name
          </label>
          <div className="mt-1">
            <input
              id="lastName"
              type="text"
              autoComplete="family-name"
              className="appearance-none block w-full px-3 py-2 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
              placeholder="Doe"
              {...register('lastName')}
            />
            {errors.lastName && (
              <p className="mt-1 text-sm text-error-600">{errors.lastName.message}</p>
            )}
          </div>
        </div>
      </div>

      {/* Email */}
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-foreground">
          Email Address
        </label>
        <div className="mt-1">
          <input
            id="email"
            type="email"
            autoComplete="email"
            className="appearance-none block w-full px-3 py-2 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
            placeholder="john@example.com"
            {...register('email')}
          />
          {errors.email && (
            <p className="mt-1 text-sm text-error-600">{errors.email.message}</p>
          )}
        </div>
      </div>

      {/* Password */}
      <div>
        <label htmlFor="password" className="block text-sm font-medium text-foreground">
          Password
        </label>
        <div className="mt-1 relative">
          <input
            id="password"
            type={showPassword ? 'text' : 'password'}
            autoComplete="new-password"
            className="appearance-none block w-full px-3 py-2 pr-10 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
            placeholder="Create a strong password"
            {...register('password')}
          />
          <button
            type="button"
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
            onClick={() => setShowPassword(!showPassword)}
          >
            {showPassword ? (
              <EyeOff className="h-5 w-5 text-muted-foreground" />
            ) : (
              <Eye className="h-5 w-5 text-muted-foreground" />
            )}
          </button>
          {errors.password && (
            <p className="mt-1 text-sm text-error-600">{errors.password.message}</p>
          )}
        </div>
      </div>

      {/* Confirm Password */}
      <div>
        <label htmlFor="confirmPassword" className="block text-sm font-medium text-foreground">
          Confirm Password
        </label>
        <div className="mt-1 relative">
          <input
            id="confirmPassword"
            type={showConfirmPassword ? 'text' : 'password'}
            autoComplete="new-password"
            className="appearance-none block w-full px-3 py-2 pr-10 border border-border rounded-md shadow-sm placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-stellar-500 focus:border-stellar-500 bg-background text-foreground"
            placeholder="Confirm your password"
            {...register('confirmPassword')}
          />
          <button
            type="button"
            className="absolute inset-y-0 right-0 pr-3 flex items-center"
            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
          >
            {showConfirmPassword ? (
              <EyeOff className="h-5 w-5 text-muted-foreground" />
            ) : (
              <Eye className="h-5 w-5 text-muted-foreground" />
            )}
          </button>
          {errors.confirmPassword && (
            <p className="mt-1 text-sm text-error-600">{errors.confirmPassword.message}</p>
          )}
        </div>
      </div>

      {/* Terms Agreement */}
      <div className="flex items-center">
        <input
          id="agreeToTerms"
          type="checkbox"
          className="h-4 w-4 text-stellar-600 focus:ring-stellar-500 border-border rounded"
          {...register('agreeToTerms')}
        />
        <label htmlFor="agreeToTerms" className="ml-2 block text-sm text-muted-foreground">
          I agree to the{' '}
          <a href="/terms" className="text-stellar-600 hover:text-stellar-500">
            Terms of Service
          </a>{' '}
          and{' '}
          <a href="/privacy" className="text-stellar-600 hover:text-stellar-500">
            Privacy Policy
          </a>
        </label>
      </div>
      {errors.agreeToTerms && (
        <p className="text-sm text-error-600">{errors.agreeToTerms.message}</p>
      )}

      {/* Submit Button */}
      <div>
        <button
          type="submit"
          disabled={isLoading}
          className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-stellar-600 hover:bg-stellar-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-stellar-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Creating Account...
            </>
          ) : (
            'Create Account'
          )}
        </button>
      </div>
    </form>
  );
} 