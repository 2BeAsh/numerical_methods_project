using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class playerHealth : MonoBehaviour

{
    public AudioSource audioPlayer_fireball_gotHit;
    public AudioSource you_dead;
    Animator animator;
    
    public GameObject[] hearts;
    public int health = 4;
    private bool alive = true;

    private void Start()
    {
        animator = GetComponent<Animator>();
    }

    private void Update()
    {
        if (health<1)
        {
            Destroy(hearts[0].gameObject);
        }
        else if (health < 2)
        {
            Destroy(hearts[1].gameObject);
        }
        else if (health < 3)
        {
            Destroy(hearts[2].gameObject);
        }
        else if (health < 4)
        {
            Destroy(hearts[3].gameObject);
        }
    }



    public void TakeDamage(int damage)
    {
        if (alive == true)
        {
            health -= damage;
            audioPlayer_fireball_gotHit.Play();
            if (health <= 0)
            {
                Die();
                alive = false;
            }
        }
    }

    void Die()
    {
        if (alive == true)
        {
            animator.SetTrigger("death");
            you_dead.Play();
        }
    }
}
