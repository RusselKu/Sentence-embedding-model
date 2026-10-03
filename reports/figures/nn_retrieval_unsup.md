### Nearest-neighbour retrieval: unsup

Corpus: 2552 unique STS-B test sentences. Over the 338 pairs with human score >= 4: Recall@1 = 76.63%, Recall@5 = 94.67%, median partner rank = 1.

**Query:** A man is playing a guitar.

| Rank | Neighbour | Cosine |
|---:|---|---:|
| 1 | A man is playing guitar. | 0.978 |
| 2 | A man is playing the guitar. | 0.961 |
| 3 | A man is playing his guitar. | 0.949 |
| 4 | A man is playing an acoustic guitar. | 0.907 |
| 5 | A man plays a guitar. | 0.904 |

**Query:** A woman is slicing an onion.

| Rank | Neighbour | Cosine |
|---:|---|---:|
| 1 | A woman is cutting an onion. | 0.966 |
| 2 | A woman is cutting onion. | 0.946 |
| 3 | A man is slicing an onion. | 0.923 |
| 4 | A person is slicing an onion. | 0.905 |
| 5 | A man is cutting an onion. | 0.885 |

**Query:** Gunmen kill nine people in northwest Pakistan

| Rank | Neighbour | Cosine |
|---:|---|---:|
| 1 | Gunmen kill nine in southwest Pakistan: Police | 0.927 |
| 2 | Gunmen kill 3 policemen in Iraq | 0.871 |
| 3 | Suicide bomber kills 21 in NW Pakistan | 0.848 |
| 4 | Gunmen kill 5 female teachers in Pakistan | 0.824 |
| 5 | Suicide attacks kill 24 people in Baghdad | 0.823 |

**Query:** The stock market fell sharply on Monday.

| Rank | Neighbour | Cosine |
|---:|---|---:|
| 1 | In other markets, U.S. Treasuries started off on Monday weaker, as stocks rose early. | 0.724 |
| 2 | Shares of McDonald's and Wendy's continued their recent recovery Monday, rising more than 1 percent on the New York Stock Exchange in afternoon trade. | 0.716 |
| 3 | Shares of McDonald's Corp. and Wendy's International Inc. continued a modest run-up on the New York Stock Exchange Monday. | 0.703 |
| 4 | The dollar fell as low as $1.1624 per euro from $1.1486 on Friday, and traded at $1.1594 at 10:15 a.m. in London. | 0.703 |
| 5 | The Swedish central bank was also meeting on Wednesday and widely expected to announce a cut on Thursday. | 0.695 |

**Worst failures (high human score, partner ranked low):**

| Human | Sentence 1 | Partner (rank, cos) | Retrieved top-1 instead (cos) |
|---:|---|---|---|
| 4.0 | There are two things to consider: | There are two possible causes for this: (87, 0.475) | I think there are two important things to consider: (0.908) |
| 4.0 | A woman is slicing some tofu. | A woman is cutting a block of tofu into small cubes. (30, 0.664) | A woman is cutting tofu. (0.908) |
| 4.0 | A brown dog is running through the field. | a brown dog with his tongue wagging as he runs through a field (16, 0.624) | A brown and white dog is running across a brown field. (0.784) |
| 4.0 | It's not a good idea. | No, it's not a good thing. (13, 0.739) | It is not a good idea. (0.931) |
| 4.0 | There are a few things I think you should do. | There are a few minimally-effective things you can do at the personal level. (13, 0.657) | There are a few things you need to consider: (0.821) |
