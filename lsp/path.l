
# A xector is a set of pairs between curly braces.  It creates a data
# structure that can be operated on in parallel.

{(sky . blue) (apple . red) (grass . green)}

# Infinte xectors contain no pairs, instead they have a few special
# forms to describe them.  The first examples is a constant xector,
# where any key maps to the value:

{'any . 1}

# lazy xectors have their value computed when accessed:

{'lazy . sqrt}

# and the universal xector maps any key to any value:

{.}

# Xectors are parallel data structures.  You can think of each key in
# a pair to be a label to a virtual processor.  If two zectors have
# the same key, then those keys values reside on the same processor,
# this is how data is distributed across processors and operated on in
# parallel.

(*cons
   {(a . 1) (b . 2) (c . 3) (d . 4) (f . 5)}
   {(b . 6) (d . 7) (e . 8) (f . 9)} )

*(cons
    ${(a . 1) (b . 2) (c . 3) (d . 4) (f . 5)}
    ${(b . 6) (d . 7) (e . 8) (f . 9)} )


{(b . (2 . 6)) (d . (4 . 7)) (f . (5 . 9))}


(struct node
   (int label)
   (xector friends) )


(def path-length ((node me) (node you))
   (let ((my-friends (get 'friends me))
         (my-friends-friends *(get 'friends $my-friends)) )
      *(set 'label $my-friends (inc (&min *(get 'label $my-friends-friends))))
      (unless (< (get 'label you) 'inf)
         *(path-length $my-friends you) )
      (get 'label you) ) )


(def histogram ((zector x))
   (&+ x '{('any . 1)}) )


(histogram '[1 2 5 4 2 4 6 7 9 9 6 5 3 2 1 2 3 4 5 4 6 7 8 9 9 0])


(main
   (setq graph {....})
   (setq me ...)
   (setq you ...)
   (print (path-length me you)) )
