# Bằng chứng và phân tích lỗi LAB 04

W4 trang 12 yêu cầu sinh viên tự chọn nguyên nhân lỗi. Bảng tự động dưới đây chỉ xuất bằng chứng, chưa phải phân tích hoàn chỉnh.

## Phân tích do sinh viên viết

- Mẫu sai 4: trích đoạn quyết định, nguyên nhân và bằng chứng: …
- Mẫu sai 5: trích đoạn quyết định, nguyên nhân và bằng chứng: …
- Mẫu sai 6: trích đoạn quyết định, nguyên nhân và bằng chứng: …
- So sánh với 3 mẫu đúng: …
- Failure quan trọng nhất và phép thử tiếp theo: …

Các loại lỗi gợi ý trong W4: lexical ambiguity, negation, sarcasm, short text, rare word, domain term, data noise, class imbalance. Không gán loại lỗi nếu chưa có bằng chứng trong văn bản/model.

<!-- AUTO:BEGIN -->
Nguồn: output thực tế trong `classification.ipynb`, phần ví dụ phân loại đúng/sai. Số liệu không thay đổi.

Model: TF-IDF unigram + Logistic Regression; test. Lấy 3 mẫu đúng và 3 mẫu sai đầu tiên theo thứ tự test, không đại diện toàn bộ loại lỗi.

| Mẫu | Kết quả | True | Predicted | P(negative) | P(positive) |
| --- | --- | --- | --- | --- | --- |
| 1 | correct | negative | negative | 0.821256848202318 | 0.178743151797682 |
| 2 | correct | negative | negative | 0.993977239468427 | 0.006022760531573 |
| 3 | correct | negative | negative | 0.858248825626068 | 0.141751174373932 |
| 4 | incorrect | positive | negative | 0.610842832810197 | 0.389157167189803 |
| 5 | incorrect | positive | negative | 0.705467451152794 | 0.294532548847206 |
| 6 | incorrect | positive | negative | 0.725158564668053 | 0.274841435331947 |

## Mẫu 1 — đúng

True=`negative`; predicted=`negative`; score lớp dự đoán=0.821256848202318.

**Văn bản thực tế đầy đủ:**

> So im not a big fan of Boll's work but then again not many are. I enjoyed his movie Postal (maybe im the only one). Boll apparently bought the rights to use Far Cry long ago even before the game itself was even finsished. <br /><br />People who have enjoyed killing mercs and infiltrating secret research labs located on a tropical island should be warned, that this is not Far Cry... This is something Mr Boll have schemed together along with his legion of schmucks.. Feeling loneley on the set Mr Boll invites three of his countrymen to play with. These players go by the names of Til Schweiger, Udo Kier and Ralf Moeller.<br /><br />Three names that actually have made them selfs pretty big in the movie biz. So the tale goes like this, Jack Carver played by Til Schweiger (yes Carver is German all hail the bratwurst eating dudes!!) However I find that Tils acting in this movie is pretty badass.. People have complained about how he's not really staying true to the whole Carver agenda but we only saw carver in a first person perspective so we don't really know what he looked like when he was kicking a**.. <br /><br />However, the storyline in this film is beyond demented. We see the evil mad scientist Dr. Krieger played by Udo Kier, making Genetically-Mutated-soldiers or GMS as they are called. Performing his top-secret research on an island that reminds me of "SPOILER" Vancouver for some reason. Thats right no palm trees here. Instead we got some nice rich lumberjack-woods. We haven't even gone FAR before I started to CRY (mehehe) I cannot go on any more.. If you wanna stay true to Bolls shenanigans then go and see this movie you will not be disappointed it delivers the true Boll experience, meaning most of it will suck.<br /><br />There are some things worth mentioning that would imply that Boll did a good work on some areas of the film such as some nice boat and fighting scenes. Until the whole cromed/albino GMS squad enters the scene and everything just makes me laugh.. The movie Far Cry reeks of scheisse (that's poop for you simpletons) from a fa,r if you wanna take a wiff go ahead.. BTW Carver gets a very annoying sidekick who makes you wanna shoot him the first three minutes he's on screen.

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| boll | 0.267984025718198 |
| far | 0.229863050009851 |
| even | 0.194178683081661 |
| not | 0.177030827890292 |
| annoying | 0.155345814243673 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.

## Mẫu 2 — đúng

True=`negative`; predicted=`negative`; score lớp dự đoán=0.993977239468427.

**Văn bản thực tế đầy đủ:**

> I had the terrible misfortune of having to view this "b-movie" in it's entirety.<br /><br />All I have to say is--- save your time and money!!! This has got to be the worst b-movie of all time, it shouldn't even be called a b-movie, more like an f-movie! Because it fails in all aspects that make a good movie: the story is not interesting at all, all of the actors are paper-thin and not at all believable, it has bad direction and the action sequences are so fake it's almost funny.......almost.<br /><br />The movie is just packed full of crappy one-liners that no respectable person could find amusing in the least little bit.<br /><br />This movie is supposed to be geared towards men, but all the women in it are SO utterly unattractive, especially that old wrinkled thing that comes in towards the end. They try to appear sexy in those weird, horrible costumes and they fail miserably!!!<br /><br />Even some of the most ridiculous b-movies will still give you some laughs, but this is just too painful to watch!!

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| worst | 0.688936059425764 |
| terrible | 0.463521595523083 |
| bad | 0.407254305081653 |
| fails | 0.402227663464193 |
| horrible | 0.363135103715643 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.

## Mẫu 3 — đúng

True=`negative`; predicted=`negative`; score lớp dự đoán=0.858248825626068.

**Văn bản thực tế đầy đủ:**

> I'm not sure why the producers needed to trade on the name of a somewhat successful movie franchise because the title suggests that it is a sequel to the first three movies..which it is not. Even though Marques Houston did appear in "HP3", he played a totally different character (he was, eight years older) in this film. Okay...so Reid and Martin weren't the most talented and couldn't carry a film all by themselves..but to trade on the HP franchise seems to me that there could have made some sort of reference (albeit minor) to the earlier movies. I'm sure everyone who wanted to see it was "hoodwinked"--into thinking that they were seeing a sequel--not a totally different film with a familiar name. And I'm sorry...Kym Whitley is not funny, and could not hold a candle to the late Robin Harris, Ketty Lester, or DC Curry in the earlier films. Although Meagen Good and Mari Morrow are a substantial visual diversion...I have to give this a THUMBS DOWN!! Just naming the film "Down to the Last Minute" would have been OK. Furthermore, the Hudlin brothers (who produced the first three movies) were not involved in the making of this film.

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| not | 0.389656741901937 |
| sorry | 0.229855074625425 |
| franchise | 0.207571207469331 |
| sequel | 0.179201797540063 |
| couldn't | 0.178346106844892 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.

## Mẫu 4 — sai

True=`positive`; predicted=`negative`; score lớp dự đoán=0.610842832810197.

**Văn bản thực tế đầy đủ:**

> The three main characters are all hopeless, and yet you only feel sorry for one of them: Ernesto, hopelessly devoted to Mercedes. This was part of the frustration: screaming at Mercedes to get a clue and ditch the no-good Harry, to no avail.<br /><br />Then there's the satisfaction: Steve Buscemi has a great part as a transvestite, and Harvey Keitel's moving story of his indignity playing a gorilla for a cheap TV movie is incredible. When you least expect it, Quentin Tarintino is doing half a monologue, and Anthony Quinn turns Ernesto into a wealthy man.<br /><br />Time and again great moments appear in the story, but in the end it's hard to know what to feel about this movie. It doesn't have a happy ending, or even a complete one, but it somehow feels right.<br /><br />This movie is strange, but then so am I; no wonder I liked it.

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| no | 0.503534317427858 |
| cheap | 0.286379340701008 |
| sorry | 0.262643258192962 |
| wonder | 0.216737591556065 |
| half | 0.204551286066403 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.

## Mẫu 5 — sai

True=`positive`; predicted=`negative`; score lớp dự đoán=0.705467451152794.

**Văn bản thực tế đầy đủ:**

> This is an OK adaptation of the breath taking book of Dan Brown. I can't say it is novel or very good but they made a movie that you can enjoy. Given the excellent story, the result could have been better though. The movie is pretty long but at the end I was feeling like some things were missing. Sound effects and sound tracks were very good. Acting was well done but the character development phase was very weak. For people who didn't read the book, things may look happening too quickly. From my point of view, instead of trying to put as much as stuff from the book, they could have tried to do the important scenes more proper. What makes the book very good was all the puzzle like story combined with the excellent portrait of Vatican. You see neither of it in the movie. Too much rush and using the time not in a good way, these are main problems of the movie. So, it is worth watching but could have been done better.

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| book | 0.559874184905710 |
| weak | 0.303918025903412 |
| instead | 0.282579820015920 |
| could | 0.251431713400615 |
| was | 0.224447137007629 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.

## Mẫu 6 — sai

True=`positive`; predicted=`negative`; score lớp dự đoán=0.725158564668053.

**Văn bản thực tế đầy đủ:**

> A fey story of a Martian attempt to colonize Earth. (Things must be pretty bad back on Mars.) Two state troopers investigate the scene of a reported UFO crash. Whatever landed is buried under the ice at Tracy's Pond but there are footsteps in the snow leading to a nearby diner.<br /><br />The diner has had no customers since eleven o'clock that morning. Now there are a handful of bus passengers sitting around waiting for permission to cross a structurally weak bridge. The bus driver insists that six passengers were aboard the bus, although he didn't notice who they were. The problem is that there are now SEVEN people waiting for the journey to be resumed. One of them is an alien, but which one? All of them are suspect. There's the crazy old man (Jack Elam), of course, who seems to exercise a sub rosa wit. There's a blustering businessman who must get to Boston (John Hoyt). A young couple on their honeymoon. (Execrable performance by the husband, Ron Kipling.) Except for the couples, nobody has noticed anyone else. And even the couples are suspicious of each other. Bride to newly minted husband: "I could have sworn you had a mole on your chin." The story continues in a sprightly but slightly spooky way -- the phone rings for no reason, the lights go on and off, the juke box turns itself on -- and none of it is to be taken seriously.<br /><br />It's a thoroughly enjoyable ensemble play and the climactic revelation is worth a chuckle. There is no discernible "depth" to it. It's not a moral message about pod people masquerading as normal citizens. It's not a warning of any kind, just a fairy tale that diverts and amuses.<br /><br />I always enjoy it when it's on. It's especially interesting to see John Hoyt as the irritable and impatient businessman, knowing that in 1954 he was the Roman Senator who masterminded the assassination of Julius Caesar in MGM's version of Shakespeare's play. And here he is -- with three arms.<br /><br />Oops.

**Top-5 feature hỗ trợ lớp dự đoán (signed coefficient × TF-IDF):**

| Feature | Contribution |
| --- | --- |
| no | 0.278289660972870 |
| bad | 0.221999810041466 |
| weak | 0.156247596038478 |
| attempt | 0.133502336351594 |
| to | 0.116957036478926 |

Đây là contribution vào logit theo hướng lớp dự đoán, không phải xác suất; chỉ 5 feature nên tổng không được coi là toàn logit.
<!-- AUTO:END -->
