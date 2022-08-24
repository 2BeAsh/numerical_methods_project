using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Prey : MonoBehaviour
{
    Animator animator;
    public AudioSource audioPlayerCollide;
    public AudioSource audioPlayerDeath;
    public ParticleSystem bloodParticleSys;
    public bool once = true;


    public float Health
    {
        set
        {
            health = value;
            if (health <= 0)
            {
                Defeated();
            }
        }
        get
        {
            return health;
        }

    }

    public float health = 1;
    public int scoreIncreaseAmount;

    public void Start()
    {
        animator = GetComponent<Animator>();
    }


    public void Defeated()
    {
        bloodParticleEffect();
        scoreCounter.instance.increaseScore(scoreIncreaseAmount);
        audioPlayerDeath.Play();
         
        animator.SetTrigger("Defeated");

    }

    public void RemoveEnemy()
    {
        Destroy(gameObject);
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        if(collision.gameObject.tag == "Player")
        {
            audioPlayerCollide.Play();
        }
    }


    public void bloodParticleEffect()
    {
        var em = bloodParticleSys.emission;
        em.enabled = true;
        bloodParticleSys.Play();
        once = false;
    }

}
