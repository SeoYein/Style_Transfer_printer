# Style_Transfer_printer
 style transfer 프린터 조작 기능 추가 

Exco 전시 

< 환경 및 실행 방법 >
- 가상환경 : artgan
- 실행파일 cam.py
 - 절대경로 입력 말고 cd Desktop/EXCO_CAM 으로 이동해서 python cam.py를 실행해야 오류가 나지 않음 
- 마우스가 아닌 키보드로만 작동
- 실행 시 느려질 경우 존재 : esc키로 창 닫고 다시 실행
- x버튼을 누를 경우 닫히지 않으니 esc로 닫아야 함 

< 기능 > 
1. stop ('s'키 누를 시 실행) 
: 화면 일시정지 기능
: 's'를 한 번 더 누를 경우 일시정지 상태가 풀리고 다시 영상 화면으로 돌아옴 

	1-2. stop after (stop 상태에서 'a'키 누를 시 실행)
	: 멈춰진 화면에서 다음 테마 적용 결과를 보여줌

	1-3. stop before(stop 상태에서 'b'키 누를 시 실행)
	: 멈춰진 화면에서 이전 테마 적용 결과를 보여줌 

	1-4. stop print(stop 상태에서 'p'키 누를 시 실행)
	: 멈춰진 화면 결과 출력 


2. after('a'키) / before('b'키)
: 영상이 재생되는 중에 테마 전환 (이전/이후)

3. print('p'키)
: 해당 상태에서 style transfer 한 결과만을 인쇄 
: 결과 사진만을 인쇄
: ./print 폴더에 들어가면 print.png (인쇄된 파일)을 볼 수 있음
: 프린트 시마다 저장이 되지 않고 파일이 갱신 

4. capture('c'키)
: print하지 않고 저장만 하는 기능
: capture_'횟수' 라는 이름으로 c누른 횟수만큼 사진이 저장 
: stop이 아닌 상태에서만 작동 

5. break('esc'키) 
: 창을 닫는 기능 
: stop이 아닌 상태에서만 작동 
