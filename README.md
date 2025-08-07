## 🚀 Git 브랜치 및 커밋 컨벤션

### 📁 브랜치 전략

```
main
├── develop
│   ├── feat/feature-name
│   ├── fix/bug-description
│   ├── chore/task-name
│   └── hotfix/urgent-fix
```

| 브랜치명     | 설명                      |
| ------------ | ------------------------- |
| `main`       | 배포용 브랜치 (운영 환경) |
| `develop`    | 개발 브랜치 (기능 통합)   |
| `feat/*`     | 새로운 기능 개발          |
| `fix/*`      | 버그 수정                 |
| `chore/*`    | 설정/빌드 작업 등         |
| `hotfix/*`   | 운영 긴급 수정            |
| `refactor/*` | 코드 리팩토링             |
| `test/*`     | 테스트 코드 작업          |
| `docs/*`     | 문서 작성 및 수정         |

---

### 💬 커밋 메시지 컨벤션

#### ✅ 형식

```
<type>: <subject>
```

#### ✅ 예시

```
feat: 로그인 페이지 UI 구현
fix: 비밀번호 입력 시 마스킹 오류 수정
docs: README에 Git 규칙 추가
refactor: useAuth 훅 구조 개선
chore: eslint 설정 파일 수정
```

#### ✅ 커밋 타입

| 타입       | 설명                                  |
| ---------- | ------------------------------------- |
| `feat`     | 새로운 기능 추가                      |
| `fix`      | 버그 수정                             |
| `docs`     | 문서 수정                             |
| `style`    | 코드 포맷 수정, 세미콜론 누락 등      |
| `refactor` | 코드 리팩토링                         |
| `test`     | 테스트 코드 작성                      |
| `chore`    | 빌드 설정, 패키지 매니저 등 기타 작업 |
| `perf`     | 성능 개선                             |
| `ci`       | CI 설정 변경                          |

---

### 📌 커밋 작성 예시

```bash
git commit -m "feat: 사용자 프로필 카드 UI 구현"
git commit -m "fix: API 호출 시 500 에러 해결"
git commit -m "refactor: 불필요한 useEffect 제거"
```
