#include <iostream>
#include <string>
using namespace std;

string function(string s, string r) {
	if(s.size()>0) {
		function(s.substr(0,s.size()-1),r+s[s.size()-1]);
	}
	return r;
}

int main(int argc, char* argv[])
{   string s="String" ;

	string r=function(s,"");

	cout << r;

	return 0;
};
