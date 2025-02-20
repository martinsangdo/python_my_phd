# Bulletproofs Protocol Implementation

This code implements both the **Prover** and **Verifier** sides of a Bulletproofs-like commitment, which is a non-interactive zero-knowledge argument of knowledge of an opening for a generalized Pedersen commitment to a vector of length n (based on the hardness of discrete logarithms).

## Key Features:
- **Commitment to a Vector**: The Bulletproof protocol allows the Prover to commit to a vector of values `x` without revealing them. This commitment is then verified by the Verifier without revealing the individual elements of the vector.
- **Groups and Hardness of Discrete Logarithm**: The implementation works on groups with chosen prime numbers `p` and `q`, where `q = 2p + 1`, ensuring that the discrete logarithm problem is hard in the subgroup of size `p`.
- **Commitment Process**: The Prover commits to a vector using a multi-round protocol that involves vector exponentiation. The commitment is updated iteratively, and the final commitment is verified by the Verifier using cryptographic hash functions and modular arithmetic.
- **Security**: The protocol ensures the integrity of the commitment process by using cryptographic hash functions and modular arithmetic to guarantee that the prover cannot alter their commitment during the protocol.

## Functions:
- **`vector_exponent(g, x, q)`**: Computes the product of `(g_i ^ x_i) mod q` for a list of generators `g` and exponents `x`.
- **`get_hash(s, p)`**: Computes a cryptographic hash of the string `s` modulo `p`, ensuring that the hash is non-zero.
- **`Prover.commit(g, x)`**: Implements the Bulletproof commitment process for the Prover, committing to a vector `x` of values using the generators `g` and returning the initial commitment, `vL` and `vR` vectors, and the final commitment value.
- **`Verifier.verify_commitment(g, com_string)`**: Verifies the commitment by checking the final commitment value against the hash and modular arithmetic conditions.

## Helper Functions:
- **`create_public_params(generator, p, q, n)`**: Generates a public set of parameters (vectors) based on a generator for the group of order `q`.
- **`create_random_vector(p, n)`**: Generates a random vector of size `n` with elements drawn from `{1, ..., p - 1}`.
- **`is_generator(g, q, p)`**: Verifies if a number `g` is a valid generator for a subgroup of size `p` in a larger group of size `q`.
- **`find_generator(q, p)`**: Finds a generator for a subgroup of size `p` in a group of size `q`.

## Tests:
The implementation includes three tests to validate the commitment process:

1. **Small Test (Correct Commitment)**: Verifies that a correctly generated commitment to a vector can be verified.
2. **Small Test (Corrupted Commitment)**: Tests the failure case where the commitment is altered, resulting in a verification failure.
3. **Large Test (Correct Commitment)**: Verifies a commitment to a larger vector, demonstrating the scalability of the protocol.

### Test Scenarios:
```python
assert(small_test_correct_commitment() == True)
assert(small_test_corrupted_commitment() == False)
assert(large_test_correct_commitment() == True)
```

## Usage:
- Set the values of `p` and `q` to the desired prime numbers, where `q = 2p + 1`.
- The `Prover` class generates commitments to a vector of values `x`, and the `Verifier` class verifies the validity of those commitments.
- Modify the `g` and `x` vectors as needed, ensuring that `g` is generated from a valid subgroup generator of size `p`.

## References:
The basic concepts of the Bulletproof protocol were referenced in the article by Dr. Anca Nitulescu. For further reading, you can explore the detailed explanations provided in [this link](https://hackmd.io/7wcukChhRwKJwXoG2T-mkA).
