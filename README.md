# 개발 및 연구 내용

## 구현 내용 상세

### IA(Information Architecture)

![AI 영양사.png](https://ifh.cc/g/FdO5Ow.png)

### Architecture

![Service Architecture light.drawio.png](https://ifh.cc/g/MHg7Co.png)

### Skills

![YOLO.drawio.png](https://ifh.cc/g/FwxlcN.png)

- AI: Tensorflow, YOLO V3
- Web: Django
- DBMS: postgreSQL
- Tools: figma, Discord, notion, github

### Database modeling

![Untitled](https://ifh.cc/g/9DaHxO.jpg)

# 실행 계획

## 기간내 프로젝트 구현 완성을 위한 전략

데이터 선정 및 EDA, 모델검토 등 단계별 소요기간 및 프로젝트 발표회를 위한 PT 준비까지의 전체 일정을 위한 Task를 계획

## 프로젝트 마일스톤

| 항목 구분 | 목표기간 | 구분 | 세부내용 |
| --- | --- | --- | --- |
| 기본 UI 설계 | 01.22 ~ 02.02 | UX | Key feature에 따른 화면정의서 |
| 기능 분석 | 01.27 ~ 01.30 | web | 구현 내용 분석, 기능 일정 수립 |
| 데이터 수집 & 가공 (전처리) | 01.27 ~ 02.07 | AI | 모델 학습에 필요한 데이터셋 수집 & 가공 (프로젝트 기간동안 계속 병행) |
| 음식 칼로리표 선정하기 | 01.27 ~ 02.04 | AI | 이미지 서칭의 기반이 될 음식 칼로리표 선정을 위해서 칼로리표 찾아보기 |
| 데이터 베이스 설계 | 01.30 ~ 01.31 | web | 유저 및 식단 관련 데이터 베이스 erd 작성 |
| 로고&컬러 선정 | 02.02 ~ 02.04 | UX | 서비스 로고 선정, 메인 및 서브 컬러 선정 |
| 로그인, 회원가입 | 02.02 ~ 02.07 | web | 카카오톡 로그인 api 연동 |
| 식단 관리 기능 | 02.02 ~ 02.11 | web | 식단 추가, 식단 편집, 일일 권장 칼로리 계산 로직, 캘린더, 통계, 사진 기능 |
| 디자인시스템 설정 | 02.05 ~ 02.06 | UX | 네이밍 규칙 정의, 아이콘 제작, 버튼 및 폰트 크기 설게 |
| 마이페이지 | 02.05 ~ 02.09 | web | 약관, 화면 구성, 사용자 데이터, 사용자 정보 입력 |
| 메인 UI 설계 및 전달 | 02.06 ~ 02.12 | UX | 디자인 파일 정리, 케이스 도출에 따른 페이지 설계 |
| 모델 API 구현 | 02.05 ~ 02.17 | AI | Django에서 Request 받고 보내는 부분은 해결 완료, Pretrained-YOLO Model 빌드해야함 |
| API Serving 구현 | 02.05 ~ 02.17 | AI | BentoML로 API값 만들기 |
| 식단 추천 기능 | 02.12 ~ 02.18 | web | 식단 추천 화면 구성, 추천 로직 구현 |
| 화면 구성 | 02.12 ~ 02.18 | web | 화면 구현, 로직과 화면 매핑 |
| 발표자료 작성 | 02.13 ~ 02.22 | UX | 발표자료 작성, 포스터 작성 |
| 기능 QA | 02.17 ~ 02.22 | UX | 구현 기능 확인 및 검수 |

## 역할 분배

| 주요 담당업무 | 역할 상세 | 인원 |
| --- | --- | --- |
| AI | 식단 사진에서 object detection task연구 및 serving | 김대현, 배성율 |
