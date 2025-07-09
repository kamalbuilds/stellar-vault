#![cfg(test)]

use super::*;
use soroban_sdk::{testutils::Address as _, Address, Env, String};

#[test]
fn test_initialize_contract() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    assert_eq!(client.get_admin(), admin);
}

#[test]
#[should_panic(expected = "Contract already initialized")]
fn test_initialize_twice_should_panic() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    client.initialize(&admin); // Should panic
}

#[test]
fn test_tokenize_real_estate_asset() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::RealEstate,
        name: String::from_str(&env, "Luxury Apartment NYC"),
        description: String::from_str(&env, "Premium apartment in Manhattan"),
        valuation: 1_000_000i128 * 10_000_000, // $1M in stroops
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner.clone(),
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Pending,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "New York, NY, USA"),
    };
    
    let total_supply = 1_000_000i128; // 1M tokens
    let token_name = String::from_str(&env, "NYC Apartment Token");
    let token_symbol = String::from_str(&env, "NYCAT");
    
    let token_address = client.tokenize_asset(
        &asset_metadata,
        &total_supply,
        &token_name,
        &token_symbol,
    );
    
    // Verify asset was stored correctly
    let stored_asset = client.get_asset(&String::from_str(&env, "ASSET_001"));
    assert_eq!(stored_asset.name, asset_metadata.name);
    assert_eq!(stored_asset.asset_type, AssetType::RealEstate);
    assert_eq!(stored_asset.valuation, asset_metadata.valuation);
    assert_eq!(stored_asset.owner, asset_owner);
    assert_eq!(stored_asset.tokenized_amount, total_supply);
    
    // Verify token contract was stored
    let stored_token_contract = client.get_token_contract(&String::from_str(&env, "ASSET_001"));
    assert_eq!(stored_token_contract, token_address);
}

#[test]
fn test_tokenize_commodities_asset() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, "GOLD_001"),
        asset_type: AssetType::Commodities,
        name: String::from_str(&env, "Gold Bars 100oz"),
        description: String::from_str(&env, "LBMA certified gold bars"),
        valuation: 200_000i128 * 10_000_000, // $200K in stroops
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner.clone(),
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Verified,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "London Vault"),
    };
    
    let total_supply = 100_000i128; // 100K tokens
    let token_name = String::from_str(&env, "Gold Token");
    let token_symbol = String::from_str(&env, "GOLD");
    
    client.tokenize_asset(
        &asset_metadata,
        &total_supply,
        &token_name,
        &token_symbol,
    );
    
    // Verify asset was stored with the provided asset_id
    let stored_asset = client.get_asset(&String::from_str(&env, "GOLD_001"));
    assert_eq!(stored_asset.asset_id, String::from_str(&env, "GOLD_001"));
    assert_eq!(stored_asset.asset_type, AssetType::Commodities);
    assert_eq!(stored_asset.compliance_status, ComplianceStatus::Verified);
}

#[test]
fn test_update_valuation() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    let ai_engine = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // First tokenize an asset
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::RealEstate,
        name: String::from_str(&env, "Test Property"),
        description: String::from_str(&env, "Test description"),
        valuation: 500_000i128 * 10_000_000,
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner,
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Pending,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "Test Location"),
    };
    
    client.tokenize_asset(
        &asset_metadata,
        &1_000_000i128,
        &String::from_str(&env, "Test Token"),
        &String::from_str(&env, "TEST"),
    );
    
    let asset_id = String::from_str(&env, "ASSET_001");
    let new_valuation = 600_000i128 * 10_000_000; // $600K
    
    // Update valuation through AI engine
    client.update_valuation(&asset_id, &new_valuation, &ai_engine);
    
    // Verify valuation was updated
    let updated_valuation = client.get_valuation(&asset_id);
    assert_eq!(updated_valuation, new_valuation);
    
    let updated_asset = client.get_asset(&asset_id);
    assert_eq!(updated_asset.valuation, new_valuation);
}

#[test]
fn test_record_transaction() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    let buyer = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // Tokenize an asset first
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::ArtCollectibles,
        name: String::from_str(&env, "Rare Painting"),
        description: String::from_str(&env, "18th century masterpiece"),
        valuation: 2_000_000i128 * 10_000_000,
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner.clone(),
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Verified,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "Private Gallery"),
    };
    
    client.tokenize_asset(
        &asset_metadata,
        &10_000i128,
        &String::from_str(&env, "Art Token"),
        &String::from_str(&env, "ART"),
    );
    
    let asset_id = String::from_str(&env, "ASSET_001");
    let amount = 100i128;
    let price_per_token = 200i128 * 10_000_000; // $200 per token
    
    // Record a transaction
    client.record_transaction(
        &asset_id,
        &asset_owner,
        &buyer,
        &amount,
        &price_per_token,
    );
    
    // Verify transaction was recorded
    let transactions = client.get_transactions(&asset_id);
    assert_eq!(transactions.len(), 1);
    
    let tx = transactions.get(0).unwrap();
    assert_eq!(tx.asset_id, asset_id);
    assert_eq!(tx.from, asset_owner);
    assert_eq!(tx.to, buyer);
    assert_eq!(tx.amount, amount);
    assert_eq!(tx.price_per_token, price_per_token);
    assert_eq!(tx.compliance_checked, true);
}

#[test]
fn test_update_compliance_status() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    let compliance_contract = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // Tokenize an asset
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::Bonds,
        name: String::from_str(&env, "Corporate Bond"),
        description: String::from_str(&env, "AAA-rated corporate bond"),
        valuation: 100_000i128 * 10_000_000,
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner,
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Pending,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "Digital Asset"),
    };
    
    client.tokenize_asset(
        &asset_metadata,
        &100_000i128,
        &String::from_str(&env, "Bond Token"),
        &String::from_str(&env, "BOND"),
    );
    
    let asset_id = String::from_str(&env, "ASSET_001");
    
    // Update compliance status
    client.update_compliance_status(
        &asset_id,
        &ComplianceStatus::Verified,
        &compliance_contract,
    );
    
    // Verify compliance status was updated
    let updated_asset = client.get_asset(&asset_id);
    assert_eq!(updated_asset.compliance_status, ComplianceStatus::Verified);
}

#[test]
fn test_transfer_ownership() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let original_owner = Address::generate(&env);
    let new_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // Tokenize an asset
    let asset_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::Infrastructure,
        name: String::from_str(&env, "Solar Farm"),
        description: String::from_str(&env, "100MW solar energy facility"),
        valuation: 50_000_000i128 * 10_000_000,
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: original_owner.clone(),
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Verified,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "California, USA"),
    };
    
    client.tokenize_asset(
        &asset_metadata,
        &50_000_000i128,
        &String::from_str(&env, "Solar Token"),
        &String::from_str(&env, "SOLAR"),
    );
    
    let asset_id = String::from_str(&env, "ASSET_001");
    
    // Transfer ownership
    client.transfer_ownership(&asset_id, &new_owner);
    
    // Verify ownership was transferred
    let updated_asset = client.get_asset(&asset_id);
    assert_eq!(updated_asset.owner, new_owner);
}

#[test]
fn test_admin_functions() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let new_admin = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // Test updating admin
    client.update_admin(&new_admin);
    assert_eq!(client.get_admin(), new_admin);
    
    // Test pause/resume contract
    client.pause_contract();
    client.resume_contract();
}

#[test]
#[should_panic(expected = "Asset name cannot be empty")]
fn test_validate_empty_name() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    let invalid_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::RealEstate,
        name: String::from_str(&env, ""), // Empty name
        description: String::from_str(&env, "Test"),
        valuation: 100_000i128,
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner,
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Pending,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "Test"),
    };
    
    client.tokenize_asset(
        &invalid_metadata,
        &1000i128,
        &String::from_str(&env, "Test"),
        &String::from_str(&env, "TEST"),
    );
}

#[test]
#[should_panic(expected = "Asset valuation must be positive")]
fn test_validate_negative_valuation() {
    let env = Env::default();
    env.mock_all_auths();
    
    let admin = Address::generate(&env);
    let asset_owner = Address::generate(&env);
    let custodian = Address::generate(&env);
    
    let contract_id = env.register(AssetTokenContract, ());
    let client = AssetTokenContractClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    let invalid_metadata = AssetMetadata {
        asset_id: String::from_str(&env, ""),
        asset_type: AssetType::RealEstate,
        name: String::from_str(&env, "Test Property"),
        description: String::from_str(&env, "Test"),
        valuation: -100_000i128, // Negative valuation
        currency: String::from_str(&env, "USD"),
        tokenized_amount: 0,
        owner: asset_owner,
        custodian: custodian,
        created_at: 0,
        compliance_status: ComplianceStatus::Pending,
        documents: Vec::new(&env),
        geographical_location: String::from_str(&env, "Test"),
    };
    
    client.tokenize_asset(
        &invalid_metadata,
        &1000i128,
        &String::from_str(&env, "Test"),
        &String::from_str(&env, "TEST"),
    );
} 