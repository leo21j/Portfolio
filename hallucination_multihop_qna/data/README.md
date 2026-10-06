# Data

The HotpotQA files are **not committed** — they
are published downloads rather than anything this project produced.

## Download

```bash
cd data
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
```

Official downloads and mirrors are listed at <https://hotpotqa.github.io/>.

The paths are configured in `configs/default.yaml` under `data:`, so you can
point them elsewhere instead of downloading here.

| File | Size | Split |
|---|---|---|
| `hotpot_dev_distractor_v1.json` | ~54 MB | Dev, distractor setting (10 passages per question, 2 gold) |
| `hotpot_dev_fullwiki_v1.json` | ~54 MB | Dev, full-wiki setting (retrieve from all of Wikipedia) |
| `hotpot_test_fullwiki_v1.json` | ~53 MB | Test, full-wiki (no answer labels) |

The pipeline reads `dev_distractor` by default. Only the dev splits carry
answers and supporting-fact annotations, so evaluation uses those.

## Format

Each file is a JSON array. The fields this project reads:

| Field | Meaning |
|---|---|
| `_id` | Question identifier, used as the prediction key |
| `question` | The multi-hop question |
| `answer` | Gold answer string (dev only) |
| `type` | `bridge` or `comparison` — selects the retriever's alpha weight |
| `level` | `easy`, `medium`, or `hard` |
| `context` | List of `[title, [sentence, ...]]` candidate passages |
| `supporting_facts` | List of `[title, sentence_index]` gold evidence (dev only) |

`pipeline/data_loader.py` parses these into `Passage`, `Context`, and
`HotpotQAExample` dataclasses.

## Citation and license

HotpotQA is distributed under CC BY-SA 4.0.

> Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen,
> Ruslan Salakhutdinov, Christopher D. Manning.
> *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*
> EMNLP 2018. <https://arxiv.org/abs/1809.09600>
