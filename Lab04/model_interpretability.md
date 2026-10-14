# Kiểm tra trọng số mô hình

## Sinh viên tự diễn giải

Feature nào phù hợp sentiment và feature nào cần kiểm tra ngữ cảnh? Dẫn review thực tế: …

<!-- AUTO:BEGIN -->
Nguồn: `run_summary.json`; SHA-256 `4eb949321de177c8ad7040d22a2529c1e4bfe9ee98d37f7a4fc0e3a5b710f4e7`. Số liệu từ output đã chạy, không fit lại hoặc sửa score.

Model baseline; label mapping `{'negative': 0, 'positive': 1}`, classes `['negative', 'positive']`. Coefficient dương hướng về positive, âm hướng về negative; là hệ số logit, không phải xác suất.

## Top positive features

| Feature | Coefficient |
| --- | --- |
| great | 8.923027790727097 |
| excellent | 7.031771211867654 |
| 7 | 5.933094009669933 |
| best | 5.640698237479344 |
| perfect | 4.86893310883196 |
| wonderful | 4.812894156700426 |
| amazing | 4.5553567210547365 |
| 8 | 4.424239558513576 |
| fun | 4.2843438485480005 |
| highly | 4.165098895107641 |
| loved | 4.160127850930879 |
| favorite | 4.158019376191622 |
| brilliant | 4.139672283670333 |
| enjoyed | 4.1003266020010125 |
| hilarious | 4.056461098676522 |
| well | 4.045224342740269 |
| today | 3.985276155483135 |
| fantastic | 3.925698966235109 |
| love | 3.9096883706083543 |
| superb | 3.8618430327167594 |

## Top negative features

| Feature | Coefficient |
| --- | --- |
| worst | -10.71114462312812 |
| bad | -8.825485057832509 |
| awful | -7.568967410442338 |
| waste | -7.43395699568513 |
| boring | -6.890276237499751 |
| poor | -6.274108022660775 |
| terrible | -6.273196047247195 |
| nothing | -5.489417348663778 |
| dull | -5.098786711057978 |
| worse | -4.968730893721106 |
| poorly | -4.726845075716762 |
| horrible | -4.627804919227703 |
| disappointing | -4.449473063642208 |
| fails | -4.429438896233701 |
| unfortunately | -4.414404352003079 |
| minutes | -4.395458256551504 |
| stupid | -4.322894479944814 |
| no | -4.257926044511551 |
| supposed | -4.1035291672502225 |
| script | -4.093714190325898 |

## Feature bigram

| Feature | Coefficient |
| --- | --- |
| not good | -1.1059921847827268 |
| very good | 1.7498258978386758 |

Bigram thuộc model unigram+bigram riêng; không so coefficient tuyệt đối giữa hai model như cùng một thang ảnh hưởng.
<!-- AUTO:END -->
