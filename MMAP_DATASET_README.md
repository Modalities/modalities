# Indexed Dataset

## Index

* doc_idx: array starts with 0; index of documents in sizes object; is one element longer so that we can use it for inclusive slicing instead of exclusive slicing
* sizes: array; number of tokens for each document
* pointers: array starts with 0; array of pointers pointing to the end of each document, whereas these positions are relative to zero and not to the start of each document

Example with 3 Documents:
doc 0: ["a", "b", "c"]
doc 1: ["d", "e", "f", "g"]
doc 2: ["h", "i"]

* doc_idx: [0, 1, 2, 3]
* sizes: [3, 4, 2]
* pointers: [0, 3, 7, 9]

When accessing an element from the index:

```python
def __getitem__(self, i):
    return self._pointers[i], self._sizes[i]
```
we get back the end of the last document and the size of the current document i.e. the offeset and how much to read next

## Sample Index File

First part of the file is an "index" with the following information stored:

* Magic HDR indicator (b'MMIDIDX\x00\x00') - indicates correct binary file
* 8 byte: version (integer)
* 1 byte: dtype of tokens in docs
* 8 byte: number of sizes == number of docs stored in file
* 8 byte: number of docs stored in file + 1
* sizes in int32
* pointers in int64
* doc_idx in int64

## Reading the index

The index is read when the Index is initialized.

```python
self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
self._bin_buffer = memoryview(self._bin_buffer_mmap)
self._sizes = np.frombuffer(
    self._bin_buffer,
    dtype=np.int32,
    count=self._len,
    offset=offset
)
```
First, `np.memmap` creates a numpy memory view, which is array-like but not a proper `np.ndarray`. Hence, with `memoryview` the `np.memmap` view is transformed in a python buffer. This python buffer can then be read by np.frombuffer, to get a proper `np.ndarray`


### Cread Sample_idx

Example

```python
if __name__ == "__main__":
    sizes = [20,50,60,30,100,5]
    _build_sample_idx(
        sizes=sizes,
        doc_idx=[0,1,2,3,4,5],
        seq_length=30,
        num_epochs=1,
        tokens_per_epoch=sum(sizes)
    )
```

Content of `sample_idx`:

```python
array([[ 0,  0],
       [ 1, 10],
       [ 1, 40],
       [ 2, 20],
       [ 2, 50],
       [ 3, 20],
       [ 4, 20],
       [ 4, 50],
       [ 4, 80]], dtype=int32)

```

### Shuffle idx

The last epoch is shuffeld seperatley to avoid undersampling of specific samples
E.g. one epoch: [1,2,3,4] train for 2.5 Epochs
* 3 Epochs: `[1,2,3,4, 1,2,3,4, 1,2,3,4] --> [3,2,1,2,3,1,1,2,3,4,4,4] --> cutoff: [3,2,1,2,3,1,1,2,3,4]`
* with this alternative impl. sample "4" would be undersampled!
* better (as is implemented in OBMD): `[1,2,3,4, 1,2,3,4,] --> [3,2,4,1,4,2,3,1] + [2,3]`

### build doc idx

We might want to refactor this method, as the recursion is always stopped on the first level and hence not really needed. A helper function should do it as well and is more readable.

```python
def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))
```


# Fine-tuning Datasets

## Instruction Tuning
Datasets, such as Bactrian or LIMA, come in different formats. Before instruction-tuning a model with one of these datasets the user has to 
transform the dataset into the following format JSONL, inspired by Fast Chat. The listing below showcases an exemplary sample from the JSONL file. 
The `id` represents the incremental sample id. `Conversations` contains the multi-turn messages between different parties. Here, we depicted messages 
between a human and a gpt model. Finally, the format allows for the specification of further, arbitrary key-value pairs such as instructions and roles.

```JSON
{
    "id": 0,
    "conversations": [
      {
        "from": "human",
        "value": "What is up?"
      },
      {
        "from": "gpt",
        "value": "Hello! How can I help you today?"
      },
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "You can call me Vicuna, and I was trained by Large Model Systems Organization (LMSYS) researchers as a language model."
      },
      {
        "from": "human",
        "value": "Goodbye"
      },
      {
        "from": "gpt",
        "value": "Goodbye! If you have any more questions in the future, don't hesitate to ask."
      }
    ]
    
    # optional / arbitrary key value pairs e.g.:
    "instruction": "Role: Vicuna, trained by Large Model Systems Organization (LMSYS) researchers"
    "role": "Vicuna, trained by Large Model Systems Organization (LMSYS) researchers"
}
```

During the instantiation of the MemMap file, we specify the JQ patterns that determine which fields in the JSON are supposed to be tokenized and additionally pass a list of special tokens e.g., `<s>`, `</s>`, `<eod>` etc. to the constructor. 
Each one of the special tokens is mapped to a single, individual token id once during the instantation of the MemMap file. 

When the dataloader iterates over the MemMap file, the `__get_item__()` method tokenizes the sample as specified in the JQ patterns list and enriches the resulting dictionary with the token ids of the special tokens. 

The dataloader packs multiple samples to a `DatasetBatch` and calls the `Collator` for bringing the samples in the correct format training. 

The collator is instantiated with information on how to assemble the entire prompt from the `conversations` and the optional key-value pairs. 
In practice, the YAML configuration has the following structure

```YAML
special_tokens:
    bos_token: <s>
    eos_token: </s>

loss_masking_jq_patterns:
    - .conversations | select(.from == "human")
    - .instruction
    - .role

message_construction: [role, instruction, conversations]
```
