# pythonStudy
pytorch : https://wikidocs.net/52460 내용 학습

git remote add origin https://github.com/SeolMuah/pythonStudy.git
git pull origin main --allow-unrelated-histories
git push -u origin master

- GitHub에 저장소 먼저 생성함

- 아래 명령 실행은 VS Code에서 commit 가능!!
git add 파일명
git commit -m "커맨트"

- 아래 명령은 직접 실행함
git remote add origin 본인의 Github 레파지토리 URL
git push origin master

* 자주 쓰는 Git 명령어
git status 
git diff
git show
git shortlog
git log


*리포지토리 변경
>git remote remove origin
1. git remote remove origin 이미 설정되어 있는 repository를 삭제

git remote add origin <링크>
2. 새로 연결하고자 하는 링크 추가

git pull origin master
3. repository에 있는 내용 받기

git commit -m “commit_description” 
4. commit하기 이때 commit_description에는 commit할 내용이 들어감 

git push origin master
