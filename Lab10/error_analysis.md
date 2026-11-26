# LAB 10 — Error Analysis (mục 24, 25, 22, 23, 14)

**Model phân tích:** `bert-base-cased` + head start/end tự viết, `max_length=384`, stride 128, seed 42 (`runs/ft_bert-base-cased_len384_lr3e-05_ep2_s42`), test = SQuAD v1.1 dev chính thức (10,570 câu, 3.29 gold/câu): EM = 0.7522, F1 = 0.8381. Baseline (keyword overlap → câu → span): EM 0.2281, F1 0.2943. Số câu EM = 0 của model: **2,619** (24.8%).

## 1. Phân loại lỗi & số lần wrong-start vs wrong-end (mục 25)

Với mỗi câu EM = 0, lấy gold khớp F1 cao nhất làm gold chính, so token đầu/cuối của prediction và gold (sau `normalize_answer`) rồi xếp vào:

- `wrong_start` — **token cuối trùng**, token đầu khác (biên trái sai);
- `wrong_end` — **token đầu trùng**, token cuối khác (biên phải sai);
- `wrong_both` — span chồng lấp gold nhưng cả hai biên đều khác;
- `wrong_location` — prediction **không chồng lấp** gold (F1 = 0: sai vị trí);
- `inner` — trùng cả hai biên theo token nhưng khác ở giữa (hiếm).

| Loại lỗi | Số câu | % của EM = 0 | % của toàn bộ test |
|---|---:|---:|---:|
| wrong_location (F1 = 0) | 970 | 37.0% | 9.2% |
| **wrong_start** | **768** | 29.3% | 7.3% |
| **wrong_end** | **640** | 24.4% | 6.1% |
| wrong_both | 221 | 8.4% | 2.1% |
| inner | 20 | 0.8% | 0.2% |
| **Tổng EM = 0** | **2,619** | 100% | 24.8% |

**#wrong start = 768, #wrong end = 640** (tỉ lệ 1.20 : 1; trong số lỗi *một biên*, wrong-start chiếm 54.5%). Đây là kết quả **ngược với dự đoán** trong `prediction.md` (tôi đã viết wrong-end nhiều hơn, 55–60%). Hệ quả đáng chú ý khác: 1,649/2,619 = **63%** lỗi vẫn chồng lấp gold (F1 > 0) và chỉ **37%** là sai vị trí hoàn toàn — phần lớn lỗi EM của model là lỗi *ranh giới*, không phải *không tìm thấy đáp án*.

Quan hệ giữa prediction và gold trong các lỗi một biên:

| | pred ⊃ gold (thừa, over-extended) | pred ⊂ gold (thiếu, partial) | lệch/chéo (shift) |
|---|---:|---:|---:|
| wrong_start (768) | 414 (54%) | 316 (41%) | 38 (5%) |
| wrong_end (640) | 381 (60%) | 234 (37%) | 25 (4%) |

Ở cả hai biên, lỗi **thừa** (prediction dài hơn gold) nhiều hơn lỗi thiếu: 795 vs 550 trên 1,408 lỗi một biên (56%). Độ dài trung bình khớp với điều đó ở gold ngắn: gold 1 từ → prediction 1.77 từ; gold 2–3 từ (2.35) → 2.84; nhưng gold 7+ từ (10.4) → prediction chỉ 6.1 từ (model *cắt ngắn* các đáp án dài; xem Exp 5 ở notebook).

Token ở biên bị sai (đếm trên 2,799 biên wrong_start và 2,801 biên wrong_end, gồm token thừa/thiếu): wrong_start 36.4% là từ chức năng (`of`×152, `and`×124, `to`×92, `in`×75, `by`, `as`, `that`); wrong_end 31.8% (`and`×156, `of`×137, `in`×81, `to`×65, `as`, `by`, `or`, `for`). Chỉ 3.6% / 2.7% là số.

### Giả thuyết cho việc wrong_start > wrong_end (chưa kiểm chứng can thiệp)

1. *Điểm vào của cụm nhiều hơn điểm ra.* Một cụm có thể bắt đầu ở nhiều chỗ hợp lệ (`the right to select …` / `to select …` / `select …`; `at partial pressures more than 50 …` / `more than 50 …` / `50 …`) và gold cũng không nhất quán giữa các annotator (mẫu 3711, 10411 bên dưới). Biên phải thường dừng ở danh từ chính nên ít lựa chọn hơn — ngược với dự đoán ban đầu của tôi rằng *điểm kết thúc* mơ hồ hơn.
2. *Câu hỏi what/how dựa vào đầu cụm.* `what`/`how`/`which` chiếm đa số và có EM thấp (0.73–0.76), trong khi `when` (0.862) / `who` (0.838) cao hơn; câu hỏi `when/who` có đáp án 1–2 token nên gần như không có biên mơ hồ.
3. *Hai softmax độc lập nên một biên "đúng" không ràng buộc biên kia* — mẫu 749, 4004, 10411 cho thấy head end chọn đúng đuôi trong khi head start bị kéo về một cụm có từ trùng câu hỏi.

Cảnh báo về độ tin cậy: 768 vs 640 chỉ từ **một seed** (42). Coi 1,408 câu là độc lập thì chênh lệch này cỡ 3.4 độ lệch chuẩn (kiểm định dấu, p < 0.001), nhưng điều đó chỉ nói *cùng một model* nghiêng về wrong-start, chưa nói nghiêng đó ổn định qua seed/model khác. Phân loại dựa vào so token (không so vị trí ký tự) nên có thể xếp nhầm khi token lặp (mẫu 9053).

### Phân loại theo giới hạn của đề (mục 24)

| Nhóm | Số câu | % của EM = 0 |
|---|---:|---:|
| over-extended (pred ⊃ gold) | 857 | 32.7% |
| partial answer (pred ⊂ gold) | 580 | 22.1% |
| wrong location, cùng câu với gold | 494 | 18.9% |
| distractor (câu khác, overlap với câu hỏi ≥ câu gold) | 332 | 12.7% |
| boundary shift | 212 | 8.1% |
| wrong location, câu khác | 144 | 5.5% |

54.8% lỗi là prediction dài hơn/ngắn hơn gold nhưng chứa nhau; chỉ 12.7% là nhầm vào một câu "mồi" giống câu hỏi hơn câu gold; 18.9% là chọn sai span *trong chính câu chứa đáp án* — thường vì câu có nhiều span cùng kiểu (hai tên, hai con số).

Theo từ đầu câu hỏi (EM / F1): `when` 0.862 / 0.910; `who` 0.838 / 0.876; `in` 0.835 / 0.889; `which` 0.760 / 0.829; `what` 0.731 / 0.825; `how` 0.732 / 0.840; `where` 0.716 / 0.823. Theo kiểu câu hỏi (heuristic `question_type` trong `baseline_qa.py`), EM của model: date 0.866, person 0.830, number 0.791, place 0.740, other 0.711.

## 2. Bảng lỗi (20 mẫu, chọn cố định bằng `random.Random(42)`, không cherry-pick)

5 `wrong_start`, 5 `wrong_end`, 5 `wrong_both` (≥ 5 wrong span theo đề), 5 `wrong_location`; mỗi loại lấy ngẫu nhiên từ toàn bộ lỗi loại đó; `idx` là chỉ số câu trong SQuAD dev (thứ tự của `load_squad_raw`). Cột "điểm" = điểm span tốt nhất / điểm `[CLS]` (null) của model (so sánh giữa window chỉ gần đúng).

| # | idx | Question | Gold | Prediction | Loại · nhóm | F1 | Điểm / null | Phân tích |
|---|---:|---|---|---|---|---:|---|---|
| 1 | 749 | Who recovered Ward's fumble? | Trevathan | T. J. Ward. Ward fumbled the ball during the return, but Trevathan | wrong_start · over-extended (pred ⊃ gold) | 0.18 | 13.7 / -7.2 | **wrong start (over-extended) + multiple plausible spans.** Biên phải đúng (`Trevathan`) nhưng biên trái nhảy ngược về `T. J. Ward` ở câu trước; span 13 token đi qua ranh giới câu. Hai head độc lập: head start bị hút bởi tên riêng `Ward` (trùng từ với câu hỏi `Ward's fumble`), head end chọn đúng người; giải mã chỉ yêu cầu `end ≥ start` và ≤ 30 token nên cặp không nhất quán vẫn có điểm cao (13.69). Không có ràng buộc 'span không vượt qua ranh giới câu' hay 'span phải là một cụm danh từ'. |
| 2 | 1776 | Complexity classes are generally classified into what? | framework | complicated definitions that do not fit into this framework | wrong_start · over-extended (pred ⊃ gold) | 0.20 | 5.3 / -8.6 | **wrong start + question ambiguity.** Gold chỉ là `framework`; model lấy cả cụm `complicated definitions that do not fit into this framework`. Câu hỏi (`classified into what?`) không khớp trực tiếp với câu chứa đáp án (đáp án gần như là một bình luận, không phải câu trả lời rõ), điểm thấp (5.32): head start chọn đầu cụm danh từ dài thay vì token cuối. Câu hỏi mơ hồ + gold rất ngắn so với cụm bao quanh. |
| 3 | 4004 | What did Low suggest using Apollo 8 for instead of simply orbiting the Earth's surface? | to orbit the Moon | waste the Saturn V on another simple Earth-orbiting mission, ASPO Manager George Low suggested the bold step … | wrong_start · over-extended (pred ⊃ gold) | 0.24 | 1.6 / -10.6 | **wrong start (over-extended).** Biên phải đúng `Moon`; biên trái bắt đầu ở `waste the Saturn V…` thay vì `orbit the Moon`. Cụm `Rather than … simply Earth-orbiting` chứa các từ trùng câu hỏi (`instead of`, `Earth orbiting`) nên head start đặt xác suất vào đầu mệnh đề chứa từ trùng thay vì vào động từ sau `to`. Điểm rất thấp (1.57, so với ~10–17 của dự đoán đúng) — model không chắc; lỗi 'biên theo từ trùng'. |
| 4 | 9045 | What is the name of another type of prime here p+1 or p-1 takes a certain shape? | Fermat primes | Sophie Germain primes | wrong_start · boundary shift | 0.40 | 7.3 / 7.7 | **multiple plausible spans / gold không đầy đủ.** Câu liệt kê `Sophie Germain primes, primorial primes, Fermat primes, Mersenne primes` — tất cả đều là đáp án hợp lệ; các gold annotator chọn các mục khác nhau (3/5 chọn `primorial primes`, còn lại `Fermat primes`, `Mersenne`) và dự đoán `Sophie Germain primes` (mục đầu danh sách) hợp lý nhưng không nằm trong gold. F1 > 0 chỉ vì trùng từ `primes`. Đây cũng là câu mà điểm `[CLS]` (7.72) cao hơn span tốt nhất (7.28): model không tự tin. Lỗi do tập gold, không hẳn do model. |
| 5 | 10411 | What is used to quantify the intuitive undestanding of forces? | standard measurement scale | precise operational definitions that are consistent with direct observations and compared to a standard measu… | wrong_start · over-extended (pred ⊃ gold) | 0.33 | 10.7 / -12.2 | **wrong start + multiple plausible spans.** Câu chứa hai cụm đều có thể là đáp án: `precise operational definitions` (4/5 gold) và `standard measurement scale` (1/5 gold). Model không chọn một trong hai mà lấy một span dài nối cả hai (từ đầu cụm thứ nhất tới cuối cụm thứ hai, qua `that are consistent with direct observations and compared to a`). Vì gold khớp F1 nhất là `standard measurement scale` nên mẫu được xếp wrong-start (biên phải trùng, biên trái lệch). Start và end là hai dự đoán độc lập: mỗi biên 'đúng' với một đáp án khác nhau. |
| 6 | 1679 | How many other important people sent letters? | more than 70 | more than 70 pioneers in science and engineering, including Albert Einstein | wrong_end · over-extended (pred ⊃ gold) | 0.43 | 8.7 / -9.7 | **wrong end (over-extended).** Câu hỏi `How many` → gold chỉ là lượng từ `more than 70`; model giữ biên trái đúng nhưng kéo biên phải tới hết câu (`pioneers in science and engineering, including Albert Einstein`). Head end ưu tiên kết thúc ở cuối cụm danh từ/mệnh đề (ranh giới ngữ pháp 'tự nhiên') thay vì ở token số lượng; điển hình cho lỗi mở rộng danh từ ở biên phải. |
| 7 | 2170 | What educational study did Luther start to pursue and immediately  drop? | law | law school | wrong_end · over-extended (pred ⊃ gold) | 0.67 | 14.1 / -8.8 | **wrong end (head noun).** `law` so với `law school`: thêm danh từ chính. Câu hỏi `what educational study` có thể đáp bằng cả hai — biên 'đúng' phụ thuộc quy ước người gán nhãn; EM không dung thứ nhưng F1 = 0.667. Lỗi biên cuối do tính mơ hồ của cụm danh từ ghép. |
| 8 | 3711 | At what point does oxygen toxicity begin to happen? | at partial pressures more than 50 kilopascals | at partial pressures more than 50 kilopascals (kPa) | wrong_end · over-extended (pred ⊃ gold) | 0.93 | 2.7 / -12.8 | **wrong end (trailing parenthetical / near-correct).** Dự đoán `at partial pressures more than 50 kilopascals (kPa)`; gold khác nhau (`50 kilopascals`, `more than 50 kilopascals`, `at partial pressures more than 50 kilopascals`) — annotator tự bất đồng về biên trái; model chọn dạng dài nhất và thêm `(kPa)` ở cuối nên chỉ lệch một token (F1 = 0.933). Normalization bỏ dấu ngoặc nhưng không bỏ token `kpa`, nên EM = 0. Gần như đúng; lỗi do tiêu chuẩn biên không nhất quán. |
| 9 | 4057 | Who worked later on to find these missing magnetic tapes that had data stored on them? | Stan Lebar | Stan Lebar, who led the team that designed and built the lunar television camera at Westinghouse Electric Cor… | wrong_end · over-extended (pred ⊃ gold) | 0.18 | 14.5 / -11.1 | **wrong end (over-extended, relative clause).** Biên trái đúng (`Stan Lebar`, 4/5 gold) nhưng biên phải kéo qua mệnh đề quan hệ `who led the team that designed and built the lunar television camera at Westinghouse Electric Corp` (F1 = 0.18). Head end không dừng ở hết tên riêng mà dừng ở cuối cả mệnh đề; một annotator chọn `Nafzger` cho thấy câu còn có đáp án hợp lệ khác. Giải mã chỉ ràng buộc `end ≥ start` và ≤ 30 token nên span dài vẫn được chấp nhận (điểm 14.48 cao vì start rất chắc). |
| 10 | 9053 | For what size natural number does Bertrand's postulate hold? | n > 3 | n | wrong_end · partial answer (pred ⊂ gold) | 0.67 | 5.1 / 7.9 | **distractor (nhãn tự động xếp nhầm là wrong end).** Dự đoán `n` nằm ở một chỗ **khác** (`for any natural number n` trong câu công thức trước), không phải `n > 3`; điểm 5.07 thấp hơn điểm `[CLS]` 7.86. Phân loại tự động so token đầu/cuối sau normalize nên `n` trùng token đầu của gold và bị xếp `wrong_end`. Nguyên nhân thật: distractor có từ trùng câu hỏi (`natural number`). Đây là hạn chế của cách phân loại (không so vị trí ký tự), cần ghi nhận. |
| 11 | 581 | What followed The Late Show with Stephen Colbert? | late local programming | a special episode of The Late Show with Stephen Colbert following the game | wrong_both · boundary shift | 0.15 | 3.8 / -10.9 | **wrong span (lexical-overlap distractor).** `What followed The Late Show with Stephen Colbert?` → gold `late local programming` (sau `Following a break for`); model chọn `a special episode of The Late Show with Stephen Colbert following the game` — chính cụm có từ trùng câu hỏi (sao chép tên chương trình) chứ không phải thứ 'theo sau'. Quan hệ thời gian (`followed`) bị bỏ qua; điểm thấp (3.78). |
| 12 | 610 | Along with 10 Cloverfield Lane, what Paramount trailer appeared during the Super Bowl? | Teenage Mutant Ninja Turtles: Out of the Shadows | The Secret Life of Pets | wrong_both · boundary shift | 0.18 | 7.5 / -5.4 | **wrong span (multiple plausible spans / binding).** Câu liệt kê `Paramount paid for TMNT… and 10 Cloverfield Lane, Universal paid for The Secret Life of Pets…`. Cần gán đúng hãng (Paramount) với trailer; model chọn `The Secret Life of Pets` (thuộc Universal) — span cùng kiểu (tên phim) trong danh sách dài. Lỗi *binding* khi có nhiều span cùng kiểu. |
| 13 | 1501 | Why did Tesla want funds from Morgan? | to complete the construction of Wardenclyffe. | appeal to his Christian spirit | wrong_both · boundary shift | 0.20 | 0.1 / -10.4 | **wrong span (long why-answer, distractor).** Câu hỏi `Why` có đáp án dài, không phải thực thể (`to complete the construction of Wardenclyffe`); model chọn một mệnh đề mục đích khác (`appeal to his Christian spirit`) ở đoạn khác, điểm rất thấp (0.13). Câu hỏi why thiếu 'điểm neo' kiểu thực thể nên mọi cụm `to + V` đều là ứng viên. |
| 14 | 5754 | What Peanuts special is Halloween-themed? | It's the Great Pumpkin, Charlie Brown | A Charlie Brown Christmas | wrong_both · boundary shift | 0.50 | 6.4 / -11.9 | **wrong span (insufficient context / cần kiến thức ngoài).** `Halloween-themed` không xuất hiện trong context; muốn trả lời phải biết `It's the Great Pumpkin, Charlie Brown` là chương trình Halloween. Model chọn `A Charlie Brown Christmas` (nhắc 2 lần, gần cụm `Peanuts specials`). Context không đủ cho suy luận thuần trích xuất; các gold cũng khác nhau về biên (`…Pumpkin` / `…Pumpkin, Charlie Brown`). |
| 15 | 7013 | What right do private schools have that public schools don't? | to select their students | the right to select their students and are funded in whole or in part by charging their students tuition | wrong_both · over-extended (pred ⊃ gold) | 0.36 | 8.3 / -13.3 | **wrong boundary (both sides, coordination).** Gold `to select their students`; model `the right to select their students and are funded in whole or in part by charging their students tuition`: sai biên trái (thêm `the right`) và biên phải (kéo qua `and are funded…`). Liên từ `and` nối hai vị ngữ nên head end không thấy điểm dừng; khớp thống kê toàn dev: `and`, `of`, `to` là các token biên bị thừa/thiếu nhiều nhất. |
| 16 | 941 | How much of Warsaw's population was Jewish? | some 30% of the city | several hundred thousand | wrong_location · wrong location (cung cau) | 0.00 | 10.3 / -11.7 | **wrong span (multiple plausible spans).** `How much of Warsaw's population` → gold `some 30% of the city`, model `several hundred thousand`. Câu có đồng vị `several hundred thousand, some 30% of the city`: cả hai đều là 'số lượng dân số Do Thái'; `much of … population` gợi tỷ lệ nên gold là phần trăm, nhưng model chọn số tuyệt đối đứng trước. Hai span hợp lý trong cùng câu. |
| 17 | 2365 | When did Luther and his wife live? | The Black Cloister | moved | wrong_location · wrong location (cung cau) | 0.00 | 4.5 / -9.2 | **wrong span (question ambiguity / noisy question).** `When did Luther and his wife live?` nhưng gold là `The Black Cloister` (đúng ra là câu hỏi `where`): câu hỏi gần như bị sai/nhiễu nhãn. Model chọn `moved` (điểm thấp 4.48) — không có thời điểm hợp lệ trong câu. Lỗi dữ liệu hơn là lỗi model. |
| 18 | 2471 | What theologian differed in views about the soul from Luther? | Johann Gerhard | Franz Pieper | wrong_location · wrong location (cung cau) | 0.00 | 14.0 / -12.2 | **wrong span (same-type distractor in same sentence).** Câu có hai tên: `Franz Pieper` (người quan sát) và `Johann Gerhard` (nhà thần học mà Luther bất đồng). Model chọn tên đầu `Franz Pieper` với điểm cao (14.02): lỗi quan hệ (`differed … from the later theologians such as X`). Cùng kiểu thực thể, cùng câu, khác vai ngữ pháp; model dựa vào kiểu thực thể và vị trí hơn là cấu trúc. |
| 19 | 5926 | What year marked the lowest ratings for the drama Lost? | 2010 | 2004 | wrong_location · distractor (cau khac, overlap >= cau gold) | 0.00 | 9.1 / -7.8 | **wrong span (cross-sentence reference, distractor).** Gold `2010` ở câu trước (`by 2010`), câu sau nói `That year, … Lost … lowest-rated season since its debut in 2004`. Từ khoá câu hỏi (`lowest ratings`, `Lost`) nằm ở câu sau nên model chọn năm gần nhất (`2004`) trong câu đó. Cần giải tham chiếu `That year` → `2010` giữa hai câu; `2004` là distractor cùng kiểu số. |
| 20 | 6981 | What is a PPP also known as? | private finance initiatives (PFIs) | Public-Private Partnering | wrong_location · wrong location (cung cau) | 0.00 | 15.7 / -9.4 | **wrong span (answer = phrase copied from question).** `What is a PPP also known as?` — model trả `Public-Private Partnering`, tức **khai triển của chính PPP** (có trong câu hỏi) thay vì đối tượng của `aka` (`private finance initiatives (PFIs)`). Cụm này nằm sát từ khoá `PPPs` và overlap từ vựng mạnh với câu hỏi; head không phân biệt 'cái được hỏi' với 'cái nhắc lại'. |

### Nhận xét chung từ 20 mẫu

1. **Nhiều lỗi "EM = 0" là do gold chưa đủ hoặc mơ hồ, không hẳn model yếu**: 9045 (4 mục liệt kê đều hợp lệ, gold chỉ ghi một số), 10411 (hai cụm đáp án hợp lệ), 3711 (gold gồm cả `50 kilopascals` lẫn `at partial pressures more than 50 kilopascals`; prediction chỉ thừa `(kPa)`, F1 0.93), 2170 (`law` vs `law school`), 4057 (một annotator chọn `Nafzger`), 2365 (câu hỏi `When` nhưng gold là địa điểm `The Black Cloister`), 5754 (cần kiến thức ngoài context). Ít nhất **7/20** mẫu thuộc nhóm này. EM = 0.75 vì vậy gần với trần thực tế của SQuAD v1.1 hơn là cho thấy model còn "dốt" (người đạt khoảng EM 0.82 / F1 0.91 theo bài báo SQuAD — tôi không tự đo lại).
2. **Over-extension ở một biên là mẫu lỗi chính** (749, 1776, 4004, 1679, 4057, 7013): một head đúng, biên kia lan qua `and`, `of`, mệnh đề quan hệ. Giải mã chỉ ràng buộc `end ≥ start` và ≤ 30 token, không có ràng buộc "dừng ở ranh giới câu / cụm danh từ" — hướng sửa chưa thử.
3. **Start và end độc lập** cho ra span ghép hai đáp án (4057, 10411, 749): mỗi biên khớp một đáp án khác nhau. Phân phối cặp (start, end) chung hoặc phạt span dài là hướng chưa thử.
4. **Distractor cùng kiểu thực thể** (581, 610, 2471, 5926, 6981): model chọn span có từ trùng câu hỏi hoặc cùng kiểu số/tên nhưng sai vai ngữ pháp/quan hệ (`followed`, `Paramount` vs `Universal`, `differed … from`, `That year`). Điểm khá cao ở 2471 (14.0) và 6981 (15.7) — **confidence cao không bảo đảm đúng**; ngược lại 1501 (0.13), 4004 (1.57), 581 (3.78) có điểm thấp, nên điểm/margin thấp là tín hiệu review hữu ích (chưa đo calibration).
5. **Có câu model "không chắc" thật sự**: 9045 và 9053 có điểm `[CLS]` *cao hơn* span tốt nhất — nếu dùng ngưỡng null như SQuAD 2.0 thì model đã từ chối trả lời hai câu này (v1.1 không có câu no-answer nên vẫn lấy span).
6. **Phân loại tự động có sai số**: 9053 bị xếp `wrong_end` vì token đầu trùng (`n`) dù span đoán nằm ở chỗ khác; so token (không so vị trí ký tự) sẽ nhầm khi token lặp.

## 3. Context analysis & nhiễu (mục 22, 24 — P1, P3, P4)

**Overlap từ vựng giữa câu hỏi và *câu chứa đáp án*** (Exp 4a, test, chia nhóm theo ngưỡng overlap): thấp (0.00–0.50, n = 4,622) EM/F1 = **0.690 / 0.785**; trung bình (0.54–0.71, n = 2,492) 0.781 / 0.859; cao (0.73–1.00, n = 3,456) **0.814 / 0.894** → chênh F1 = 10.9 điểm giữa nhóm cao và thấp. Model chọn đúng câu chứa đáp án 85.3% / 96.3% / 98.2%; baseline (chỉ dựa overlap) 55.3% / 93.3% / 97.0%. Overlap thấp khó hơn vì đáp án phải tìm qua paraphrase/suy luận; ngay cả ở nhóm cao baseline chỉ EM 0.30 nên **chọn đúng câu chưa đủ** — bước định vị span trong câu mới là chỗ phân biệt.

**Bộ cặp paraphrase tự viết (12 cặp, chỉ minh hoạ, không đủ lớn để kết luận)**: cùng một đoạn, câu hỏi trùng từ vs paraphrase — FT EM/F1 = 0.750 / 0.839 vs 0.833 / 0.859 (paraphrase *không* tệ hơn, **khác dự đoán**), baseline F1 0.644 vs 0.472 (baseline kém hẳn khi paraphrase). FT sai ở: `Who wrote Hamlet?` → `Christopher Marlowe` (cả hai dạng câu hỏi), `When does the library close on weekdays?` → `6 p.m` (nhầm giờ cuối tuần). Với n = 12 thì chênh một cặp = 8 điểm EM nên không dùng làm bằng chứng.

**Thí nghiệm nhiễu** (`perturbation_results.csv`; 1,984/2,000 câu mẫu đủ điều kiện cho mọi kiểu nhiễu; ΔF1 so với context gốc, CI 95% bootstrap theo câu):

| Điều kiện | FT EM | FT F1 | ΔF1 [CI 95%] | FT pred rơi vào câu chèn | Baseline ΔF1 |
|---|---:|---:|---|---:|---:|
| gốc | 0.7535 | 0.8374 | — | — | — |
| +2 câu ngẫu nhiên (rand2) | 0.7525 | 0.8379 | +0.0005 [−0.008, +0.009] | 0.2% | −0.0005 |
| +1 câu mồi cùng kiểu (distr1) | 0.7344 | 0.8087 | **−0.0287 [−0.040, −0.019]** | **5.0%** | −0.0193 |
| nối 2 đoạn *trước* (long_pre2) | 0.7243 | 0.8080 | **−0.0294 [−0.041, −0.018]** | — | −0.0028 |
| nối 2 đoạn *sau* (long_post2) | 0.7445 | 0.8297 | −0.0077 [−0.017, +0.001] | — | 0 |
| câu đáp án lên đầu | 0.7606 | 0.8412 | +0.0037 [−0.005, +0.013] | — | +0.0057 |
| câu đáp án xuống cuối | 0.7354 | 0.8206 | −0.0168 [−0.026, −0.008] | — | −0.0095 |
| xáo trộn toàn bộ câu | 0.7414 | 0.8229 | −0.0145 [−0.024, −0.005] | — | −0.0032 |

Đối chiếu P1/P3/P4 (chi tiết ở `README.md` mục 6): câu ngẫu nhiên không ảnh hưởng (đúng); **câu mồi chỉ hạ F1 2.9 điểm và chỉ 5.0% prediction rơi vào câu mồi — nhỏ hơn nhiều so với −10 đến −25 tôi dự đoán**; baseline bị mồi lừa nhiều hơn (11.2% rơi vào câu chèn vs 5.0% của FT). Câu mồi tôi tạo (lấy từ bài khác, overlap cao, cùng kiểu) có thể *dễ phân biệt hơn* mồi do người viết đối kháng, nên 2.9 điểm là ước lượng thấp của độ nhạy với distractor. Đáp án ở cuối context (−1.7) và xáo trộn (−1.5) hại hơn đáp án ở đầu (+0.4, ≈ 0); xáo trộn nhỏ hơn nhiều so với −3 đến −6 dự đoán. Nối đoạn *trước* (2.03 window/câu) hại gấp ~4 lần nối đoạn *sau* (−2.9 vs −0.8 điểm); giả thuyết chưa kiểm chứng: điểm span giữa các window không so sánh được hoàn toàn (mỗi window softmax riêng) nên window không chứa đáp án có thể "thắng".

## 4. Experiment 1 — Context length có luôn giúp không?

`max_length` 128 / 256 / 384 (stride = ⌊max_length/3⌋ = 42 / 85 / 128, các thành phần khác cố định, seed 42):

| max_length | train features (20,000 câu) | test features | EM test | F1 test | thời gian train (s, thuần) | VRAM đỉnh | suy luận test (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 43,952 | 24,295 | 0.7473 | 0.8323 | 1,349 | 2,310 MB | 66.3 |
| 256 | 22,420 | 12,096 | 0.7497 | 0.8354 | 1,062 | 2,732 MB | 50.9 |
| 384 | 20,234 | 10,822 | 0.7522 | 0.8381 | 1,022 | 3,124 MB | 46.9 |

F1 và EM tăng đơn điệu theo context length (F1 +0.0031 từ 128→256, +0.0027 từ 256→384), **nhưng mỗi bước chỉ cỡ 1 độ lệch chuẩn giữa các seed** (std 3 seed ở len384: EM 0.0038, F1 0.0027) và 128/256 chỉ có một seed ⇒ **không kết luận được** 384 tốt hơn 128 một cách đáng tin; đây không phải bằng chứng cho "context dài hơn luôn tốt hơn". Thứ thay đổi rõ rệt là *chi phí*: context SQuAD trung bình ≈ 175 token nên 384 hầu như không cắt (1.0117 window/câu; 187 window không chứa đáp án; 0 câu mất đáp án); ở 128 mỗi câu thành 2.2 window nên số feature gấp đôi và thời gian train **lâu hơn** 32% (1,349 vs 1,022 s) dù mỗi feature ngắn hơn, đổi lại VRAM đỉnh thấp hơn 26% (2.3 vs 3.1 GB). Thời gian đo trên laptop dùng chung nên có thể bị ảnh hưởng bởi tiến trình khác; chênh 1,349 vs 1,022 s lớn hơn nhiều so với dao động giữa ba seed của len384 (1,022 / 1,073 / 1,069 s).

Theo độ dài context (token; F1 test của model 128 / 256 / 384): ≤128 (n = 2,007): 0.840 / 0.827 / 0.831; 129–256 (n = 7,267): 0.833 / 0.842 / 0.842; 257–384 (n = 1,079): 0.816 / 0.816 / 0.829; > 384 (n = 217): 0.812 / 0.800 / 0.833. Model 128 kém hơn 384 ở nhóm 257–384 (−1.3 điểm), phù hợp với lập luận "window nhỏ mất ngữ cảnh"; nhưng nhóm > 384 chỉ có 217 câu (nhiễu cỡ ±3 điểm) và model 256 lại thấp hơn 128 ở đó, nên không rút ra được quy luật ổn định.

## 5. Phép đo & giới hạn

- Chỉ **một seed** cho 128/256 và cho toàn bộ phân tích lỗi; 3 seed chỉ cho `len384` (test EM 0.7485 ± 0.0038, F1 0.8359 ± 0.0027).
- Train chỉ 20k/87.6k câu hỏi (23%), 2 epoch, không tinh chỉnh siêu tham số nên EM 0.75 thấp hơn BERT-base đầy đủ (≈ 0.81 EM / 0.88 F1 theo bài báo gốc — không tự đo).
- Phân loại lỗi so token sau normalize, không so vị trí; câu nhiều gold lấy gold khớp F1 cao nhất ⇒ có thể nhầm loại khi các annotator bất đồng.
- 20 mẫu cố định seed 42 chỉ để *minh hoạ* các nhóm lỗi, không đại diện tỉ lệ (tỉ lệ nằm ở bảng đếm mục 1).
- Phần giải thích từng mẫu do AI viết sau khi xem span; chỉ các con số (điểm, F1, độ dài) là đo được, còn "nguyên nhân" là giả thuyết chưa kiểm chứng bằng can thiệp (ví dụ xoá từ trùng câu hỏi rồi chạy lại).
