import os
import sys

 
def main(outdir, commit_tag, cesga_user):
    print(f'outdir={outdir}')
    print(f'commit_tag={commit_tag}')
    print(f'cesga_user={cesga_user}')
    with open(os.path.join(outdir,'outfile.txt'), 'w') as file:
      file.write(f'logging: outdir={outdir}\n')
      file.write(f'logging: commit_tag={commit_tag}\n')
      file.write(f'logging: cesga_user={cesga_user}\n')

if __name__ == "__main__" :
    commit_tag, cesga_user = sys.argv[1:]
    outdir = 'output'
    os.makedirs('output')
    main(outdir, commit_tag, cesga_user)

