#include <stdio.h>

template < class _T >
class C
	{
	public:
		template < class _Ta >
		void X(){ puts( "test" ); }
	};

template < class _T >
void F()
	{
	C< int > obj1;
	obj1.X<int>();

	C< _T > obj2;
	obj2.X<_T>();
	}

int main()
{
F< int >();
return 0;
}
