#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <utility>
using namespace std;

typedef pair<int, int> pii;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " IN OUT" << endl;
    exit(1);
  }

  vector<pii> edges;
  
  ifstream in(argv[1]);
  assert(in.is_open());

  string buf;
  getline(in, buf);
  while (buf[0] == '%') {
      getline(in, buf);
  }

  int n, m;
  sscanf(buf.c_str(), "%d %d %d", &n, &n, &m);
  
  edges.reserve(m);
  for (int i = 0; i < m; ++i) {
    getline(in, buf);
    istringstream parser(buf);
  
    int a, b;
    parser >> a >> b;
    edges.push_back(make_pair(a, b));
  }
  assert(edges.size() == m);

  ofstream out(argv[2], ios::binary);    
  assert(out.is_open());
  
  out.write((char*)&n, sizeof(int));
  out.write((char*)&m, sizeof(int));
  out.write((char*)edges.data(), edges.size() * sizeof(pii));
}