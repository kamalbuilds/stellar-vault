export function TechStack() {
  const technologies = [
    {
      category: 'Blockchain',
      items: [
        { name: 'Stellar', description: 'High-performance blockchain', logo: '⭐' },
        { name: 'Soroban', description: 'Smart contracts platform', logo: '🔧' },
        { name: 'Freighter', description: 'Wallet integration', logo: '💳' }
      ]
    },
    {
      category: 'AI & ML',
      items: [
        { name: 'TensorFlow', description: 'Machine learning framework', logo: '🧠' },
        { name: 'OpenAI', description: 'Large language models', logo: '🤖' },
        { name: 'Hugging Face', description: 'Model deployment', logo: '🤗' }
      ]
    },
    {
      category: 'Infrastructure',
      items: [
        { name: 'Next.js', description: 'React framework', logo: '⚡' },
        { name: 'Kubernetes', description: 'Container orchestration', logo: '☸️' },
        { name: 'PostgreSQL', description: 'Database system', logo: '🐘' }
      ]
    }
  ];

  return (
    <section className="py-16 lg:py-24 bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-bold text-foreground mb-4">
            Built with Industry-Leading Technology
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Enterprise-grade infrastructure combining the best of blockchain, AI, and modern web technologies
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {technologies.map((tech, index) => (
            <div key={index} className="text-center">
              <h3 className="text-xl font-semibold text-foreground mb-6">
                {tech.category}
              </h3>
              <div className="space-y-4">
                {tech.items.map((item, i) => (
                  <div key={i} className="p-4 bg-card border border-border rounded-lg hover:shadow-md transition-shadow">
                    <div className="flex items-center space-x-3">
                      <span className="text-2xl">{item.logo}</span>
                      <div className="text-left">
                        <div className="font-medium text-foreground">{item.name}</div>
                        <div className="text-sm text-muted-foreground">{item.description}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <div className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-stellar-600 to-vault-600 text-white rounded-lg">
            <span className="font-medium">Powered by cutting-edge technology stack</span>
          </div>
        </div>
      </div>
    </section>
  );
} 