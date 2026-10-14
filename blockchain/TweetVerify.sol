// SPDX-License-Identifier: MIT
pragma solidity ^0.8.11;

/**
 * TweetVerify — Store & Verify Sentiment Analysis Results on Blockchain
 * 
 * Purpose: Anyone can register a Merkle root hash of sentiment results.
 * Because blockchain is immutable, the hash cannot be tampered with.
 * Anyone can later verify that a given result matches the stored hash.
 */
contract TweetVerify {
    
    struct Record {
        uint id;
        string description;      // e.g., "Sentiment batch 1"
        string merkleRoot;       // SHA-256 Merkle root of the batch
        uint timestamp;
        address submitter;       // who registered this
    }
    
    // Storage
    mapping(uint => Record) public records;
    uint public recordCount = 0;
    
    // Events (for logging on blockchain)
    event RecordAdded(
        uint indexed id,
        string merkleRoot,
        address indexed submitter,
        uint timestamp
    );
    
    // ------------------------------------------------------------
    // Add a new record (register a Merkle root)
    // ------------------------------------------------------------
    function addRecord(string memory _description, string memory _merkleRoot) public {
        require(bytes(_merkleRoot).length > 0, "Merkle root cannot be empty");
        
        recordCount++;
        records[recordCount] = Record(
            recordCount,
            _description,
            _merkleRoot,
            block.timestamp,
            msg.sender
        );
        
        emit RecordAdded(recordCount, _merkleRoot, msg.sender, block.timestamp);
    }
    
    // ------------------------------------------------------------
    // Verify if a given Merkle root matches a stored record
    // ------------------------------------------------------------
    function verify(uint _id, string memory _merkleRoot) public view returns (bool) {
        require(_id > 0 && _id <= recordCount, "Invalid record ID");
        return keccak256(abi.encodePacked(records[_id].merkleRoot)) 
            == keccak256(abi.encodePacked(_merkleRoot));
    }
    
    // ------------------------------------------------------------
    // Get a record's details
    // ------------------------------------------------------------
    function getRecord(uint _id) public view returns (
        uint, string memory, string memory, uint, address
    ) {
        require(_id > 0 && _id <= recordCount, "Invalid record ID");
        Record memory r = records[_id];
        return (r.id, r.description, r.merkleRoot, r.timestamp, r.submitter);
    }
}