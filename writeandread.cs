using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace mytranslator
{
    class Program
    {
        static void Main(string[] args)
        {
             string[] lines = System.IO.File.ReadAllLines(@"D:\propertities\total10.txt");

            StreamWriter sr = File.CreateText(@"D:\propertities\ha.txt");

            foreach (string i in lines)
            {
                string[] arrs = i.Split('_');
                string tmpha = "";
                foreach(string j in arrs)
                {
                    
                    tmpha = tmpha + j;
                    tmpha = tmpha + " ";
                }
                string tmp = "Tom's "+tmpha;
                Console.WriteLine(tmp);
                sr.WriteLine(tmp);
            }
            sr.Close();
        }
    }
}
