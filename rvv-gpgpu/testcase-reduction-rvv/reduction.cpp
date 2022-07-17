#include<iostream>
#include<fstream>
using namespace std;

int main()
{
	int N; // total number of data need to be reducted
	int warp_num; // number of warp working in one SM
	int thread_num; // number of thread for one warp
	int i,j,k,m,n;
	ifstream infile;
	cin>>N>>warp_num>>thread_num;
	infile.open("data.txt");
	int *p = new int[N];
	for(i = 0 ; i<N ; i++){
		infile>>p[i];
	}
	int **sp = new int *[warp_num];
	for(n=0;n<warp_num;n++){
		sp[n] = new int [thread_num];
	}
	cout<<"first round reduction result:"<<endl;
	for(j = 0; j<warp_num ;j++){
		cout<<"warp "<<j<<":";
		for(k = 0 ; k < thread_num ; k++){
			if(k!=0){
				cout<<",";
			}
			sp[j][k] = 0 ;
			for(m = j*thread_num + k ; m < N ; m = m + warp_num*thread_num){
				sp[j][k] = sp[j][k] + p[m];
			}
			cout<<sp[j][k];
		}
		cout<<endl;
	}
	cout<<"second round reduction result:"<<endl;
	for(j = 0; j< warp_num ; j++){
		m=thread_num / 2;
		while(m!=0){
			cout<<"warp "<<j<<":";
			for( k = 0;k < m;k++){
				sp[j][k] += sp[j][k+m];
				if(k!=0){cout<<",";}
				cout<<sp[j][k];
			}
			for( k= m; k<thread_num;k++){
				cout<<", dc ";
			}
			cout<<endl;
			m /= 2;
		}
	}
	delete[] p;
	int *kp = new int[warp_num];
	for(i=0;i<warp_num;i++){
		kp[i] = sp[i][0];
	}
	cout<<"third round reduction result:"<<endl;
	cout<<"warp 0:";
	if(warp_num<=thread_num){
		m = warp_num;
		for(i=0;i<warp_num;i++){cout<<kp[i]<<",";}
		for(i=warp_num;i<thread_num;i++){cout<<" dc ,";
		}
		cout<<endl;
	}else{
		m= thread_num;
		for(j=0;j<thread_num;j++){
			for(k = j+thread_num;k<warp_num;j=j+thread_num){
			kp[j] += kp[k];
			}
			cout<<kp[j]<<",";
		}
		cout<<endl;
	}
	cout<<"forth round reduction result:"<<endl;
	m /= 2;
	while(m!=0){
			cout<<"warp 0:";
			for( k = 0;k < m;k++){
				kp[k] += kp[k+m];
				if(k!=0){cout<<",";}
				cout<<kp[k];
			}
			for( k= m; k<thread_num;k++){
				cout<<", dc ";
			}
			cout<<endl;
			m /= 2;}
	for(n=0;n<warp_num;n++)
	{
		delete[] sp[n];
	}
	delete[] sp;
	delete[] kp;
	
	infile.close();
	
}
