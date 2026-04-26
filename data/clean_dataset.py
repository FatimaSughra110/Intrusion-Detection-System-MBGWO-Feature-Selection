import pandas as pd

files = ['NetFlow_v2_Features.csv', 'NF-ToN-IoT-v2.csv']

for file in files:
    output_name = f"subset_05_{file}"
    chunks = []
    
    # Read in chunks of 100,000 rows at a time
    for chunk in pd.read_csv(file, chunksize=100000):
        # Sample 5% of the current chunk
        sampled_chunk = chunk.sample(frac=0.05, random_state=42)
        chunks.append(sampled_chunk)
    
    # Combine the sampled chunks into one final dataframe
    final_subset = pd.concat(chunks)
    final_subset.to_csv(output_name, index=False)
    print(f"Created {output_name} with {len(final_subset)} rows.")